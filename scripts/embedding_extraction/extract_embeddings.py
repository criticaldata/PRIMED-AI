"""
Extract frozen V-JEPA2 embeddings from echo MP4 videos.

Supports all V-JEPA2 model variants: ViT-L, ViT-H, ViT-g, ViT-g/384.
Loads the encoder with pretrained weights, runs each video through it,
mean-pools the patch tokens to a single embedding vector, and saves all embeddings.

Usage:
    python scripts/extract_embeddings.py --model vitl --limit 5
    python scripts/extract_embeddings.py --model vitl --folder p10 --num_workers 8
    python scripts/extract_embeddings.py --model vith
    python scripts/extract_embeddings.py --model vitg
    python scripts/extract_embeddings.py --model vitg-384
"""

import argparse
import os
import sys

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset

# src/ is bundled alongside this script in scripts/embedding_extraction/src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_base, vit_giant_xformers, vit_huge, vit_large

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

WEIGHTS_DIR = "/orcd/pool/006/lceli_shared/weights"
EMBEDDINGS_DIR = "/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo"

MODEL_REGISTRY = {
    # V-JEPA2 natural image pretrained
    "vitl": {"constructor": vit_large, "embed_dim": 1024, "img_size": 256, "batch_size": 256},
    "vith": {"constructor": vit_huge, "embed_dim": 1280, "img_size": 256, "batch_size": 128},
    "vitg": {
        "constructor": vit_giant_xformers,
        "embed_dim": 1408,
        "img_size": 256,
        "batch_size": 64,
    },
    "vitg-384": {
        "constructor": vit_giant_xformers,
        "embed_dim": 1408,
        "img_size": 384,
        "batch_size": 16,
    },
    # EchoJEPA — fine-tuned on MIMIC-IV-Echo (Alif Munim / Bo Wang lab)
    "echo-vitl-scratch": {
        "constructor": vit_large,
        "embed_dim": 1024,
        "img_size": 256,
        "batch_size": 256,
        "checkpoint": "vitl-scratch-pt-210-c25.pt",
    },
    "echo-vitl-mimic117": {
        "constructor": vit_large,
        "embed_dim": 1024,
        "img_size": 256,
        "batch_size": 256,
        "checkpoint": "vjepa21_vitl_mimic_pt117.pt",
    },
    "echo-vitb-mimic169": {
        "constructor": vit_base,
        "embed_dim": 768,
        "img_size": 256,
        "batch_size": 512,
        "checkpoint": "vjepa2_1_vitb_mimic_pt169_c60.pt",
    },
    "echo-vitl-mimic100": {
        "constructor": vit_large,
        "embed_dim": 1024,
        "img_size": 256,
        "batch_size": 256,
        "checkpoint": "vjepa21_vitl_mimic_pt100.pt",
    },
    "echo-vitl-vmix22m": {
        "constructor": vit_large,
        "embed_dim": 1024,
        "img_size": 256,
        "batch_size": 256,
        "checkpoint": "vitl-vmix22m-pt220-c55.pt",
    },
}


# ---------------------------------------------------------------------------
# Worker init (matches src/datasets/video_dataset.py:31-43)
# ---------------------------------------------------------------------------
def _worker_init_fn(_):
    """Keep each DataLoader worker to 1 CPU thread to avoid oversubscription."""
    try:
        import os as _os

        import cv2
        import torch as _torch

        _torch.set_num_threads(1)
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
        _os.environ["OMP_NUM_THREADS"] = "1"
        _os.environ["MKL_NUM_THREADS"] = "1"
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Dataset & collate for parallel video loading
# ---------------------------------------------------------------------------
class EchoVideoDataset(Dataset):
    """Map-style dataset over MP4 files for DataLoader-based parallel loading."""

    def __init__(self, file_paths, input_dir, num_frames, transform):
        self.file_paths = file_paths
        self.input_dir = input_dir
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mp4_path = self.file_paths[idx]
        rel_path = os.path.relpath(mp4_path, self.input_dir)
        try:
            video = load_video(mp4_path, self.num_frames)
            if video is None:
                return None
            video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)  # [T, C, H, W]
            video_tensor = self.transform(video_tensor)  # [C, T, H, W]
            return rel_path, video_tensor
        except Exception:
            return None


def safe_collate_fn(batch):
    """Filter out None entries from failed video loads, then stack."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    rel_paths, tensors = zip(*batch)
    return list(rel_paths), torch.stack(tensors)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Extract V-JEPA2 embeddings")
    parser.add_argument(
        "--model",
        type=str,
        default="vitl",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model variant (default: vitl)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/orcd/pool/006/lceli_shared/mimic-iv-echo-mp4",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Process single subfolder (e.g., p10). For SLURM array jobs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt weights (auto-resolved from --model)",
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="Output .pt path (auto-resolved from --model)"
    )
    parser.add_argument(
        "--img_size", type=int, default=None, help="Input resolution (auto-resolved from --model)"
    )
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Batch size for GPU inference (0 = auto per model)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader workers for parallel video loading (0 = sequential fallback)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max files to process (0 = all)")
    parser.add_argument("--save_every", type=int, default=10000, help="Checkpoint every N videos")
    args = parser.parse_args()

    # When --folder is set, scope input and output to that subfolder
    if args.folder:
        args.input_dir = os.path.join(args.input_dir, args.folder)

    # Auto-resolve defaults from model registry
    cfg = MODEL_REGISTRY[args.model]
    if args.checkpoint is None:
        ckpt_file = cfg.get("checkpoint", f"{args.model}.pt")
        args.checkpoint = os.path.join(WEIGHTS_DIR, ckpt_file)
    if args.output_path is None:
        suffix = f"_{args.folder}" if args.folder else ""
        args.output_path = os.path.join(EMBEDDINGS_DIR, f"{args.model}_embeddings{suffix}.pt")
    if args.img_size is None:
        args.img_size = cfg["img_size"]
    if args.batch_size == 0:
        args.batch_size = cfg["batch_size"]

    return args


def load_model(model_name, checkpoint_path, img_size, num_frames):
    """Load a V-JEPA2 encoder with pretrained weights."""
    cfg = MODEL_REGISTRY[model_name]
    model = cfg["constructor"](
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

    # Load weights — same pattern as notebooks/vjepa2_demo.py:load_pretrained_vjepa_pt_weights
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["encoder"]
    clean = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(clean, strict=False)
    print(f"Loaded {model_name} weights: {msg}")

    model.cuda().eval()
    return model


def build_transform(img_size):
    """Eval transform: resize, center crop, normalize (from notebooks/vjepa2_demo.py)."""
    short_side = int(256.0 / 224 * img_size)
    return video_transforms.Compose(
        [
            video_transforms.Resize(short_side, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_video(path, num_frames):
    """Load video and uniformly sample num_frames frames."""
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    total = len(vr)
    if total == 0:
        return None
    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    video = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
    return video


# ---------------------------------------------------------------------------
# Sequential fallback (original loop, for debugging with --num_workers 0)
# ---------------------------------------------------------------------------
def extract_sequential(model, transform, remaining, args, embeddings):
    """Original sequential extraction loop."""
    processed = 0
    errored = 0
    bs = args.batch_size
    total = len(remaining)

    for batch_start in range(0, total, bs):
        batch_paths = remaining[batch_start : batch_start + bs]

        batch_tensors = []
        batch_rel_paths = []
        for mp4_path in batch_paths:
            rel_path = os.path.relpath(mp4_path, args.input_dir)
            try:
                video = load_video(mp4_path, args.num_frames)
                if video is None:
                    continue
                video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)
                video_tensor = transform(video_tensor)
                batch_tensors.append(video_tensor)
                batch_rel_paths.append(rel_path)
            except Exception as e:
                errored += 1
                print(f"  ERROR loading: {rel_path} ({e})")

        if not batch_tensors:
            continue

        try:
            batch_input = torch.stack(batch_tensors).cuda()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                features = model(batch_input)
                batch_embs = features.mean(dim=1).float().cpu()

            for j, rel_path in enumerate(batch_rel_paths):
                embeddings[rel_path] = batch_embs[j]
                processed += 1

        except Exception as e:
            errored += len(batch_rel_paths)
            print(f"  ERROR batch forward: {e}")

        done = min(batch_start + bs, total)
        if done % 1000 < bs or done == total:
            print(f"  [{done}/{total}] processed={processed} errored={errored}")

        if args.save_every > 0 and processed > 0 and processed % args.save_every < bs:
            torch.save(embeddings, args.output_path)
            print(f"  Checkpoint saved: {len(embeddings)} embeddings")

    return processed, errored


# ---------------------------------------------------------------------------
# DataLoader-based extraction (parallel video loading)
# ---------------------------------------------------------------------------
def extract_dataloader(model, transform, remaining, args, embeddings):
    """Parallel extraction with DataLoader for overlapped CPU/GPU work."""
    dataset = EchoVideoDataset(remaining, args.input_dir, args.num_frames, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2,
        collate_fn=safe_collate_fn,
        worker_init_fn=_worker_init_fn,
        persistent_workers=True,
        pin_memory=True,
    )

    processed = 0
    errored = 0
    total = len(remaining)
    bs = args.batch_size

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            errored += bs
            continue

        rel_paths, batch_input = batch

        try:
            batch_input = batch_input.cuda(non_blocking=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                features = model(batch_input)  # [B, num_patches, embed_dim]
                batch_embs = features.mean(dim=1).float().cpu()  # [B, embed_dim]

            for j, rel_path in enumerate(rel_paths):
                embeddings[rel_path] = batch_embs[j]
                processed += 1

        except Exception as e:
            errored += len(rel_paths)
            print(f"  ERROR batch forward: {e}")

        done = min((batch_idx + 1) * bs, total)
        if done % 5000 < bs or done >= total:
            print(f"  [{done}/{total}] processed={processed} errored={errored}")

        if args.save_every > 0 and processed > 0 and processed % args.save_every < bs:
            torch.save(embeddings, args.output_path)
            print(f"  Checkpoint saved: {len(embeddings)} embeddings")

    return processed, errored


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = MODEL_REGISTRY[args.model]

    print(f"Model: {args.model} (embed_dim={cfg['embed_dim']}, img_size={args.img_size})")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_path}")
    if args.folder:
        print(f"Folder: {args.folder}")
    print(f"Num workers: {args.num_workers}")

    # Collect MP4 files
    mp4_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith(".mp4"):
                mp4_files.append(os.path.join(root, f))
    mp4_files.sort()

    if args.limit > 0:
        mp4_files = mp4_files[: args.limit]

    if not mp4_files:
        print(f"No MP4 files found in {args.input_dir}")
        return

    print(f"Found {len(mp4_files)} MP4 file(s)")

    # Resume: load existing checkpoint if present
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if os.path.exists(args.output_path):
        embeddings = torch.load(args.output_path, map_location="cpu", weights_only=False)
        print(f"Resumed from checkpoint: {len(embeddings)} existing embeddings")
    else:
        embeddings = {}

    # Filter out already-processed files
    remaining = []
    for mp4_path in mp4_files:
        rel_path = os.path.relpath(mp4_path, args.input_dir)
        if rel_path not in embeddings:
            remaining.append(mp4_path)

    total = len(remaining)
    skipped = len(mp4_files) - total
    if skipped > 0:
        print(f"Skipping {skipped} already-processed files, {total} remaining")

    if total == 0:
        print("All files already processed!")
        return

    # Load model and transform
    model = load_model(args.model, args.checkpoint, args.img_size, args.num_frames)
    transform = build_transform(args.img_size)

    print(f"Batch size: {args.batch_size}")

    # Extract embeddings
    if args.num_workers == 0:
        print("Mode: sequential (--num_workers 0)")
        processed, errored = extract_sequential(model, transform, remaining, args, embeddings)
    else:
        print(f"Mode: DataLoader ({args.num_workers} workers, prefetch_factor=2)")
        processed, errored = extract_dataloader(model, transform, remaining, args, embeddings)

    # Final save
    torch.save(embeddings, args.output_path)

    # Summary
    print(f"\nDone! processed={processed}, errored={errored}, total={len(embeddings)}")
    print(f"Saved {len(embeddings)} embeddings to {args.output_path}")
    if embeddings:
        all_embs = torch.stack(list(embeddings.values()))
        print(f"Shape per video: {list(embeddings.values())[0].shape}")
        print(f"Stats: mean={all_embs.mean():.4f}, std={all_embs.std():.4f}")
        print(f"All finite: {torch.isfinite(all_embs).all()}")


if __name__ == "__main__":
    main()
