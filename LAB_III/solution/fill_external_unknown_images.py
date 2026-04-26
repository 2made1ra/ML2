"""Fill ``data_lab3/external_images`` with Unknown Genre WikiArt images.

The notebook section 8 expects exactly 20 local images in
``data_lab3/external_images``. This script samples images from the local
``raw_data/wikiart_dataset`` copy where ``genre == "Unknown Genre"`` and writes
them as JPEG files. It does not train or evaluate any model.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import load_from_disk
from PIL import Image, ImageFile


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "raw_data" / "wikiart_dataset"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data_lab3" / "external_images"

UNKNOWN_CLASS_NAME = "Unknown Genre"
SUPPORTED_OUTPUT_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

FALLBACK_GENRE_NAMES = [
    "abstract_painting",
    "cityscape",
    "genre_painting",
    "illustration",
    "landscape",
    "nude_painting",
    "portrait",
    "religious_painting",
    "sketch_and_study",
    "still_life",
    UNKNOWN_CLASS_NAME,
]

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create exactly 20 external test images from the local WikiArt "
            "Unknown Genre class."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Local Hugging Face dataset saved with save_to_disk. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory used by lab_3.ipynb section 8. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of Unknown Genre images to write. Default: 20",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible sampling. Default: 42",
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=768,
        help="Resize images so their longest side is at most this value. Default: 768",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help=(
            "Do not remove existing jpg/jpeg/png/webp files first. Use this only "
            "if the directory is already empty or you intentionally want extra files."
        ),
    )
    return parser.parse_args()


def get_raw_class_names(train_split) -> list[str]:
    feature = train_split.features.get("genre")
    names = getattr(feature, "names", None)
    return list(names) if names else FALLBACK_GENRE_NAMES


def label_to_int(label, raw_class_names: list[str]) -> int:
    if isinstance(label, int):
        return label
    if isinstance(label, str):
        return raw_class_names.index(label)
    raise TypeError(f"Unsupported genre label type: {type(label)}")


def clear_supported_images(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_OUTPUT_SUFFIXES:
            path.unlink()


def save_sample_image(item, output_path: Path, max_edge: int) -> None:
    image = item["image"].convert("RGB")
    if max_edge > 0:
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    image.save(output_path, format="JPEG", quality=92)

    with Image.open(output_path) as saved:
        saved.verify()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory was not found: {args.dataset_dir}")

    dataset = load_from_disk(str(args.dataset_dir))
    train_split = dataset["train"] if "train" in dataset else dataset
    raw_class_names = get_raw_class_names(train_split)
    unknown_label = raw_class_names.index(UNKNOWN_CLASS_NAME)

    unknown_indices = [
        idx
        for idx, label in enumerate(train_split["genre"])
        if label_to_int(label, raw_class_names) == unknown_label
    ]
    if len(unknown_indices) < args.count:
        raise ValueError(
            f"Need {args.count} Unknown Genre images, found only {len(unknown_indices)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        clear_supported_images(args.output_dir)

    selected_indices = random.Random(args.seed).sample(unknown_indices, args.count)
    for out_idx, dataset_idx in enumerate(selected_indices, start=1):
        output_path = args.output_dir / f"unknown_genre_{out_idx:02d}.jpg"
        save_sample_image(train_split[dataset_idx], output_path, args.max_edge)

    final_images = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_OUTPUT_SUFFIXES
    )
    if len(final_images) != args.count:
        raise RuntimeError(
            f"Expected exactly {args.count} supported images in {args.output_dir}, "
            f"found {len(final_images)}. Remove extra images or rerun without --keep-existing."
        )

    print(f"Saved {len(final_images)} Unknown Genre images to {args.output_dir}")


if __name__ == "__main__":
    main()
