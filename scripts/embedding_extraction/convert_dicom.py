"""
Convert MIMIC-IV-Echo DICOM files to MP4 videos.

Recursively walks the input directory, converts multi-frame DICOMs to MP4 videos,
and skips single-frame (static) images.

Usage:
    python data/convert_dicom.py --limit 5                    # test with 5 files
    python data/convert_dicom.py --target_size 384            # for ViT-g/16_384
    python data/convert_dicom.py                              # convert all (256x256)
    python data/convert_dicom.py --workers 8                  # parallel with 8 workers
"""

import argparse
import os
import signal
from multiprocessing import Pool

import cv2
import numpy as np
import pydicom

# Threshold for switching to memory-safe per-frame normalization (2 GB)
_LARGE_THRESHOLD = 2 * 1024**3


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout("processing exceeded time limit")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert DICOM to MP4")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/orcd/pool/006/lceli_shared/mimic-iv-echo/files/mimic-iv-echo-0.1.physionet.org",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/orcd/pool/006/lceli_shared/mimic-iv-echo-mp4",
    )
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0, help="Max files to convert (0 = all)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (0 = all CPUs)")
    parser.add_argument("--timeout", type=int, default=600, help="Per-file timeout in seconds (default 600 = 10 min)")
    return parser.parse_args()


def process_dicom(file_path, save_path, target_size, fps):
    """Convert a single DICOM to MP4. Returns (status, reason)."""
    ds = pydicom.dcmread(file_path)

    if "PixelData" not in ds:
        return "skipped", "no pixel data"

    pixel_array = ds.pixel_array

    # Single-frame (2D) → skip
    if pixel_array.ndim == 2:
        return "skipped", "single frame"

    num_frames = pixel_array.shape[0]
    if num_frames < 2:
        return "skipped", f"only {num_frames} frame(s)"

    # Estimate decompressed size to decide processing path
    estimated_bytes = pixel_array[0].nbytes * num_frames

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    size = (target_size, target_size)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, size, isColor=True)

    if estimated_bytes > _LARGE_THRESHOLD:
        # --- Memory-safe path: normalize per-frame to avoid bulk float32 copy ---
        p_min = float(pixel_array.min())
        p_max = float(pixel_array.max())
        scale = 255.0 / (p_max - p_min) if p_max > p_min else 0.0

        for frame in pixel_array:
            if scale > 0:
                frame = ((frame.astype(np.float32) - p_min) * scale).astype(np.uint8)
            else:
                frame = np.zeros_like(frame, dtype=np.uint8)
            resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            if resized.ndim == 2:
                bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            else:
                bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            out.write(bgr)

        out.release()
        return "converted", f"{num_frames} frames (large-file path)"
    else:
        # --- Original path (unchanged) ---
        pixel_array = pixel_array.astype(np.float32)
        p_min, p_max = pixel_array.min(), pixel_array.max()
        if p_max - p_min > 0:
            pixel_array = ((pixel_array - p_min) / (p_max - p_min)) * 255.0
        else:
            pixel_array = np.zeros_like(pixel_array)
        pixel_array = pixel_array.astype(np.uint8)

        for frame in pixel_array:
            resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            if resized.ndim == 2:
                bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            else:
                bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            out.write(bgr)

        out.release()
        return "converted", f"{num_frames} frames"


def _convert_one(task):
    """Worker function for multiprocessing. Returns (rel_path, status, reason)."""
    dcm_path, save_path, rel_path, target_size, fps = task

    # Skip files that already have a converted output
    if os.path.exists(save_path):
        return rel_path, "skipped", "already exists"

    # 5-minute timeout per file so one bad DICOM can't block a worker
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(300)
    try:
        status, reason = process_dicom(dcm_path, save_path, target_size, fps)
        signal.alarm(0)
        return rel_path, status, reason
    except _Timeout:
        signal.alarm(0)
        # Clean up partial output
        if os.path.exists(save_path):
            os.remove(save_path)
        return rel_path, "error", "timeout (5min)"
    except Exception as e:
        signal.alarm(0)
        return rel_path, "error", str(e)
    finally:
        signal.signal(signal.SIGALRM, old_handler)


def main():
    args = parse_args()

    # Recursively collect all .dcm files
    dcm_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith(".dcm"):
                dcm_files.append(os.path.join(root, f))
    dcm_files.sort()

    if args.limit > 0:
        dcm_files = dcm_files[: args.limit]

    total = len(dcm_files)
    print(f"Found {total} DICOM file(s) to process")
    print(f"Target size: {args.target_size}x{args.target_size}")
    print(f"Workers: {args.workers if args.workers >= 1 else 'all CPUs'}")

    # Build task list
    tasks = []
    for dcm_path in dcm_files:
        rel_path = os.path.relpath(dcm_path, args.input_dir)
        save_path = os.path.join(args.output_dir, rel_path.replace(".dcm", ".mp4"))
        tasks.append((dcm_path, save_path, rel_path, args.target_size, args.fps))

    converted = 0
    skipped = 0
    errored = 0

    if args.workers == 1:
        # Sequential path with reliable signal-based timeout (works in main process)
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        for i, (dcm_path, save_path, rel_path, target_size, fps) in enumerate(tasks):
            if os.path.exists(save_path):
                skipped += 1
                if (i + 1) % 1000 == 0 or (i + 1) == total:
                    print(f"  [{i+1}/{total}] converted={converted} skipped={skipped} errored={errored}")
                continue
            signal.alarm(args.timeout)
            try:
                status, reason = process_dicom(dcm_path, save_path, target_size, fps)
                signal.alarm(0)
                if status == "converted":
                    converted += 1
                    print(f"  [{i+1}/{total}] OK: {rel_path} ({reason})")
                else:
                    skipped += 1
                    print(f"  [{i+1}/{total}] SKIP: {rel_path} ({reason})")
            except _Timeout:
                signal.alarm(0)
                if os.path.exists(save_path):
                    os.remove(save_path)
                errored += 1
                print(f"  [{i+1}/{total}] TIMEOUT: {rel_path} (>{args.timeout}s)")
            except Exception as e:
                signal.alarm(0)
                errored += 1
                print(f"  [{i+1}/{total}] ERROR: {rel_path} ({e})")
        signal.signal(signal.SIGALRM, old_handler)
    else:
        # Parallel path
        num_workers = args.workers if args.workers >= 1 else None
        with Pool(num_workers) as pool:
            for i, (rel_path, status, reason) in enumerate(
                pool.imap_unordered(_convert_one, tasks, chunksize=1)
            ):
                if status == "converted":
                    converted += 1
                elif status == "error":
                    errored += 1
                else:
                    skipped += 1
                if (i + 1) % 1000 == 0 or (i + 1) == total:
                    print(f"  [{i+1}/{total}] converted={converted} skipped={skipped} errored={errored}")

    print(f"\nDone! converted={converted}, skipped={skipped}, errored={errored}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
