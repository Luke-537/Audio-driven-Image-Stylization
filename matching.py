import argparse
import csv
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def list_files(root: Path, exts: set, limit: int, seed: int) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    if not files:
        return []
    random.Random(seed).shuffle(files)
    return files[:limit]


def greedy_max_match(sim: torch.Tensor) -> List[Tuple[int, int, float]]:
    """
    Greedy 1:1 matching on similarity matrix.
    Returns list of (image_idx, audio_idx, score).
    """
    n_i, n_a = sim.shape
    flat_scores = sim.flatten()
    # Sort all pairs by descending score
    sorted_idx = torch.argsort(flat_scores, descending=True)
    used_i = set()
    used_a = set()
    matches = []
    for idx in sorted_idx.tolist():
        i = idx // n_a
        a = idx % n_a
        if i in used_i or a in used_a:
            continue
        score = float(sim[i, a].item())
        matches.append((i, a, score))
        used_i.add(i)
        used_a.add(a)
        if len(matches) >= min(n_i, n_a):
            break
    return matches


def main():
    parser = argparse.ArgumentParser(description="Match images and audio using ImageBind embeddings.")
    parser.add_argument("--images-dir", type=str, default="datasets/images_small", help="Root folder with images (recursively scanned).")
    parser.add_argument("--audio-dir", type=str, default="datasets/audio_subset", help="Root folder with audio (recursively scanned).")
    parser.add_argument("--num-samples", type=int, default=20, help="Max samples for each modality.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--output-dir", type=str, default="outputs/matches", help="Where to save CSV/JSON (and copies if --copy).")
    parser.add_argument("--copy", action="store_true", help="Copy matched image+audio files into output-dir for quick inspection.")
    args = parser.parse_args()

    images_root = Path(args.images_dir)
    audio_root = Path(args.audio_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect files
    image_paths = list_files(images_root, IMAGE_EXTS, args.num_samples, args.seed)
    audio_paths = list_files(audio_root, AUDIO_EXTS, args.num_samples, args.seed)

    if not image_paths:
        print(f"No images found in: {images_root} (extensions: {sorted(IMAGE_EXTS)})")
        return
    if not audio_paths:
        print(f"No audio files found in: {audio_root} (extensions: {sorted(AUDIO_EXTS)})")
        return

    print(f"Found {len(image_paths)} images and {len(audio_paths)} audio files.")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    # Prepare inputs
    try:
        vis_inputs = data.load_and_transform_vision_data([str(p) for p in image_paths], device)
        aud_inputs = data.load_and_transform_audio_data([str(p) for p in audio_paths], device)
    except Exception as e:
        print("Failed to load/transform media. If this is an audio DLL/FFmpeg issue, ensure FFmpeg and VC++ Redistributables are installed.")
        print(f"Error: {e}")
        return

    inputs = {
        ModalityType.VISION: vis_inputs,
        ModalityType.AUDIO: aud_inputs,
    }

    with torch.no_grad():
        embeds = model(inputs)

    # Normalize (cosine similarity)
    v = embeds[ModalityType.VISION]
    a = embeds[ModalityType.AUDIO]
    v = torch.nn.functional.normalize(v, dim=1)
    a = torch.nn.functional.normalize(a, dim=1)
    sim = v @ a.T  # (N_img, N_audio)

    # Greedy 1:1 matching
    matches = greedy_max_match(sim)

    # Save results
    ts = time.strftime("%Y%m%d-%H%M%S")
    csv_path = out_root / f"audio_image_matches_{ts}.csv"
    json_path = out_root / f"audio_image_matches_{ts}.json"

    rows: List[Dict] = []
    for rank, (i_idx, a_idx, score) in enumerate(matches, start=1):
        row = {
            "rank": rank,
            "image_path": str(image_paths[i_idx]),
            "audio_path": str(audio_paths[a_idx]),
            "score": round(score, 6),
            "image_rel": str(image_paths[i_idx].relative_to(images_root)) if images_root in image_paths[i_idx].parents else image_paths[i_idx].name,
            "audio_rel": str(audio_paths[a_idx].relative_to(audio_root)) if audio_root in audio_paths[a_idx].parents else audio_paths[a_idx].name,
        }
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"\nSaved matches:")
    print(f"- CSV : {csv_path}")
    print(f"- JSON: {json_path}")

    # Optional: copy matched files into folders for quick eyeballing
    if args.copy:
        for rank, row in enumerate(rows, start=1):
            pair_dir = out_root / f"pair_{rank:04d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            # Preserve original file names
            dst_img = pair_dir / Path(row["image_path"]).name
            dst_aud = pair_dir / Path(row["audio_path"]).name
            try:
                shutil.copy2(row["image_path"], dst_img)
                shutil.copy2(row["audio_path"], dst_aud)
            except Exception as e:
                print(f"Copy failed for pair {rank}: {e}")
        print(f"Copied matched files into subfolders under: {out_root}")

    # Print a small summary to console
    print("\nTop 10 matched pairs (image -> audio) with cosine similarity:")
    for row in rows[:10]:
        print(f"[{row['rank']:02d}] {Path(row['image_path']).name}  <->  {Path(row['audio_path']).name}   score={row['score']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()