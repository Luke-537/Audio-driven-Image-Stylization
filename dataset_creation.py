import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict

import sys
import torch
from PIL import Image, ImageFile

# Make ImageBind submodule importable: repo_root/ImageBind
_repo_root = Path(__file__).resolve().parent
_imagebind_submodule = _repo_root / "ImageBind"
if _imagebind_submodule.exists():
    sys.path.insert(0, str(_imagebind_submodule))

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


# Images (PNG/JPG/JPEG) used for visual matching
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def list_files(root: Path, exts: set) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def filter_valid_images(paths: List[Path]) -> List[Path]:
    """Return only images that PIL can open. Allows truncated images if possible.

    Counts and skips unreadable images; use this for robust pipelines.
    """
    # Allow loading of some truncated files
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    valid = []
    failed = 0
    for p in paths:
        try:
            with Image.open(p) as img:
                img.verify()  # quick integrity check
            valid.append(p)
        except Exception:
            # Skip unreadable/truncated images
            failed += 1
    if failed:
        print(f"Skipped {failed} unreadable/truncated image(s).")
    return valid


def greedy_assign_images_to_audio(sim: torch.Tensor) -> List[int]:
    """For each audio embedding (rows), pick best image (cols).

    Returns a list `best_img_idx` of length N_audio, where each entry is the
    index of the selected image.
    """

    # sim: (N_audio, N_img)
    best_img_idx = torch.argmax(sim, dim=1)  # per-row argmax
    return best_img_idx.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create an audio–image dataset using ImageBind embeddings. "
            "Each audio file is paired with its most similar image."
        )
    )

    parser.add_argument(
        "--images-dir",
        type=str,
        default="/graphics/scratch2/students/fmarvin/Stylesheets",
    )
    parser.add_argument(
        "--audio-dirs",
        type=str,
        nargs="+",
        default=["/graphics/scratch2/students/reutemann/ESC-50-master"],
        help=(
            "One or more root folders containing audio files"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used when encoding audio with ImageBind.",
    )

    args = parser.parse_args()

    images_root = Path(args.images_dir)
    audio_roots = [Path(p) for p in args.audio_dirs]
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect files
    image_paths = list_files(images_root, IMAGE_EXTS)
    image_paths = filter_valid_images(image_paths)
    audio_paths: List[Path] = []
    for r in audio_roots:
        audio_paths.extend(list_files(r, AUDIO_EXTS))

    if not image_paths:
        print(f"No image files found in: {images_root} (extensions: {sorted(IMAGE_EXTS)})")
        return
    if not audio_paths:
        print(
            "No audio files found in the provided audio dirs "
            f"(extensions: {sorted(AUDIO_EXTS)})"
        )
        return

    print(f"Found {len(image_paths)} images and {len(audio_paths)} audio files.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    # Encode all images once using VISION modality.
    print("Encoding images with ImageBind (VISION modality)...")
    with torch.no_grad():
        image_inputs = data.load_and_transform_vision_data(
            [str(p) for p in image_paths], device
        )
        img_embeds = model({ModalityType.VISION: image_inputs})[ModalityType.VISION]
        img_embeds = torch.nn.functional.normalize(img_embeds, dim=1)

    # Encode audio in batches and assign a stylesheet to each audio file.
    print("Encoding audio files and matching to images...")
    batch_size = max(1, int(args.batch_size))

    all_rows: List[Dict] = []

    with torch.no_grad():
        for start in range(0, len(audio_paths), batch_size):
            end = min(start + batch_size, len(audio_paths))
            batch_paths = audio_paths[start:end]
            print(f"Processing audio batch {start}–{end - 1} / {len(audio_paths) - 1}")

            try:
                aud_inputs = data.load_and_transform_audio_data(
                    [str(p) for p in batch_paths], device
                )
            except Exception as e:
                print(f"Failed to load/transform audio batch {start}-{end}: {e}")
                continue

            embeds = model({ModalityType.AUDIO: aud_inputs})
            a = embeds[ModalityType.AUDIO]
            a = torch.nn.functional.normalize(a, dim=1)

            # Similarity: (N_audio_batch, N_img)
            sim = a @ img_embeds.T
            best_img_idx = greedy_assign_images_to_audio(sim)

            for audio_idx_in_batch, img_idx in enumerate(best_img_idx):
                audio_path = batch_paths[audio_idx_in_batch]
                image_path = image_paths[img_idx]
                score = float(sim[audio_idx_in_batch, img_idx].item())

                row: Dict = {
                    "audio_path": str(audio_path),
                    "image_path": str(image_path),
                    "similarity": round(score, 6),
                    "audio_rel": str(audio_path),
                    "image_rel": str(image_path.relative_to(images_root))
                    if images_root in image_path.parents
                    else image_path.name,
                }
                all_rows.append(row)

    if not all_rows:
        print("No pairs were created (all batches failed?).")
        return

    # Save dataset: one row per audio file with its best image match.
    csv_path = out_root / "audio_image_pairs.csv"
    json_path = out_root / "audio_image_pairs.json"

    fieldnames = ["audio_path", "image_path", "similarity", "audio_rel", "image_rel"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    print("\nSaved dataset:")
    print(f"- CSV : {csv_path}")
    print(f"- JSON: {json_path}")


if __name__ == "__main__":
    main()

