"""Frozen EchoJEPA-L (V-JEPA2 ViT-L) encoder wrapper (M01)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from primed_ai.encoders.base import assert_frozen, freeze_module

_EXTRACTION_ROOT = Path(__file__).resolve().parents[3] / "scripts" / "embedding_extraction"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_ALIASES = {
    "echo-vitl-mimic117": "vjepa21_vitl_mimic_pt117.pt",
    "echo-vitl-scratch": "vitl-scratch-pt-210-c25.pt",
    "echo-vitb-mimic169": "vjepa2_1_vitb_mimic_pt169_c60.pt",
    "vitl": "vitl.pt",
}


def _ensure_extraction_path() -> None:
    root = str(_EXTRACTION_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_vit_encoder(
    *,
    checkpoint_path: str | Path,
    embed_dim: int,
    img_size: int,
    num_frames: int,
    model_name: str = "echo-vitl-mimic117",
) -> nn.Module:
    """Load the bundled V-JEPA2 ViT encoder and checkpoint weights."""
    _ensure_extraction_path()
    from src.models.vision_transformer import vit_base, vit_large  # noqa: WPS433

    constructors = {"vit_large": vit_large, "vit_base": vit_base}
    use_base = "vitb" in model_name or embed_dim == 768
    constructor = constructors["vit_base" if use_base else "vit_large"]
    model = constructor(
        img_size=(img_size, img_size),
        num_frames=num_frames,
        patch_size=16,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=True,
        use_rope=True,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["encoder"] if isinstance(ckpt, dict) and "encoder" in ckpt else ckpt
    clean = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
    incompatible = model.load_state_dict(clean, strict=False)
    # Unexpected keys are fine (checkpoints carry predictor/EMA tensors we don't use), but
    # missing keys mean the prefix stripping above didn't match this checkpoint layout and
    # the ViT would stay randomly initialised.
    if incompatible.missing_keys:
        raise RuntimeError(
            f"{len(incompatible.missing_keys)} encoder weights not found in {checkpoint_path} "
            f"(first: {incompatible.missing_keys[:3]}) — checkpoint key layout is unsupported"
        )
    return model


class EchoJEPALEncoder(nn.Module):
    """Frozen EchoJEPA-L encoder: ``[B, C, T, H, W]`` video -> token / pooled embeddings."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        embed_dim: int = 1024,
        img_size: int = 256,
        num_frames: int = 16,
        model_name: str = "echo-vitl-mimic117",
        frozen: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.num_frames = num_frames
        self.model_name = model_name
        self.checkpoint_path = str(checkpoint_path)
        self._backbone = _load_vit_encoder(
            checkpoint_path=checkpoint_path,
            embed_dim=embed_dim,
            img_size=img_size,
            num_frames=num_frames,
            model_name=model_name,
        )
        if frozen:
            freeze_module(self._backbone)
        self.to(device)

    @classmethod
    def from_config(cls, cfg, *, device: str | torch.device = "cpu") -> EchoJEPALEncoder:
        """Construct from a Hydra/OmegaConf encoder config node."""
        return cls(
            cfg.checkpoint_path,
            embed_dim=int(cfg.get("embed_dim", 1024)),
            img_size=int(cfg.get("img_size", 256)),
            num_frames=int(cfg.get("num_frames", 16)),
            model_name=str(cfg.get("name", "echojepa")),
            frozen=bool(cfg.get("frozen", True)),
            device=device,
        )

    def forward(
        self,
        video: torch.Tensor,
        *,
        return_tokens: bool = True,
        pool: Literal["mean", "none"] = "mean",
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass on preprocessed video tensors.

        ``video`` shape: ``[B, C, T, H, W]`` (ClipToTensor + Normalize applied upstream).
        Returns ``tokens`` ``[B, N, D]`` and optionally ``pooled`` ``[B, D]``.
        """
        with torch.set_grad_enabled(any(p.requires_grad for p in self._backbone.parameters())):
            tokens = self._backbone(video)
        out: dict[str, torch.Tensor] = {}
        if return_tokens:
            out["tokens"] = tokens
        if pool == "mean":
            out["pooled"] = tokens.mean(dim=1)
        return out

    def encode(self, video: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Alias for :meth:`forward` (pipeline API)."""
        return self.forward(video, **kwargs)

    def verify_frozen(self) -> None:
        assert_frozen(self._backbone)


def build_video_transform(img_size: int = 256):
    """Eval transform matching ``scripts/embedding_extraction/extract_embeddings.py``."""
    _ensure_extraction_path()
    import src.datasets.utils.video.transforms as video_transforms  # noqa: WPS433
    import src.datasets.utils.video.volume_transforms as volume_transforms  # noqa: WPS433

    short_side = int(256.0 / 224 * img_size)
    return video_transforms.Compose(
        [
            video_transforms.Resize(short_side, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
