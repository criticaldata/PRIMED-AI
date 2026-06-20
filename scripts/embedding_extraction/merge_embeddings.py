"""Merge per-folder embedding .pt files into a single combined file."""

import argparse
import os

import torch

EMBEDDINGS_DIR = "/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo"
FOLDERS = [f"p{i}" for i in range(10, 20)]


def main():
    parser = argparse.ArgumentParser(description="Merge per-folder embeddings")
    parser.add_argument("--model", type=str, default="vitl")
    parser.add_argument("--embeddings_dir", type=str, default=EMBEDDINGS_DIR)
    args = parser.parse_args()

    combined = {}
    for folder in FOLDERS:
        path = os.path.join(args.embeddings_dir, f"{args.model}_embeddings_{folder}.pt")
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu", weights_only=False)
            print(f"  {folder}: {len(data)} embeddings")
            combined.update(data)
        else:
            print(f"  {folder}: MISSING")

    out_path = os.path.join(args.embeddings_dir, f"{args.model}_embeddings_all.pt")
    torch.save(combined, out_path)

    all_embs = torch.stack(list(combined.values()))
    print(f"\nTotal: {len(combined)} embeddings saved to {out_path}")
    print(f"Shape: {all_embs.shape}")
    print(f"Stats: mean={all_embs.mean():.4f}, std={all_embs.std():.4f}")
    print(f"All finite: {torch.isfinite(all_embs).all()}")


if __name__ == "__main__":
    main()
