import argparse
from pathlib import Path
import shutil
import os
from typing import List
import json
import random


def split_annotations(annotations_file: dict):
    images = annotations_file["images"]

    # Create subsets based on the `split` field
    train_images = [img for img in images if img["split"] == "TRAIN"]
    val_images = [img for img in images if img["split"] == "VAL"]

    # Create subsets for the test set based on the `tag` field
    test_real_images = [
        img for img in images if img["split"] == "TEST" and img["tag"] == "REAL"
    ]
    test_id_images = [
        img for img in images if img["split"] == "TEST" and img["tag"] == "ID"
    ]
    test_ood_images = [
        img for img in images if img["split"] == "TEST" and img["tag"] == "OOD"
    ]

    # Helper function to filter annotations by image_id
    def filter_annotations(images_subset):
        image_ids = {img["id"] for img in images_subset}
        return [
            ann
            for ann in annotations_file["annotations"]
            if ann["image_id"] in image_ids
        ]

    # Create new COCO subsets
    datasets = {
        "train": {
            "info": annotations_file["info"],
            "images": train_images,
            "annotations": filter_annotations(train_images),
            "categories": annotations_file["categories"],
        },
        "val": {
            "info": annotations_file["info"],
            "images": val_images,
            "annotations": filter_annotations(val_images),
            "categories": annotations_file["categories"],
        },
        "test_real": {
            "info": annotations_file["info"],
            "images": test_real_images,
            "annotations": filter_annotations(test_real_images),
            "categories": annotations_file["categories"],
        },
        "test_id": {
            "info": annotations_file["info"],
            "images": test_id_images,
            "annotations": filter_annotations(test_id_images),
            "categories": annotations_file["categories"],
        },
        "test_ood": {
            "info": annotations_file["info"],
            "images": test_ood_images,
            "annotations": filter_annotations(test_ood_images),
            "categories": annotations_file["categories"],
        },
    }

    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-path", type=str, help="Path to the original Singapore Dataset."
    )
    parser.add_argument(
        "-tvr",
        "--train_val_test_ratio",
        type=lambda arg: arg.split(","),
        help="Target Width and Height of the dataset images.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Path where the coco version of the dataset will " "" "be saved.",
    )
    parser.add_argument(
        "-z",
        "--zip",
        type=str,
        default=None,
        help="Zip path, if None, don't zip the result folder",
    )
    args = parser.parse_args()

    train_val_test_ratio = [float(v) for v in args.train_val_test_ratio]
    assert (
        sum(train_val_test_ratio) == 1.0
    ), "Sum of train, validation and test (real, ood, id) ratio must be 1."
    output_path = Path(args.output_path)
    data_path = Path(args.data_path)
    if data_path.suffix == ".zip":
        shutil.unpack_archive(data_path, data_path.parent)
        data_path = data_path.parent / data_path.stem
    annotations_folder = os.path.join(data_path, "annotations")
    images_folder = os.path.join(data_path, "images")
    annotations_file = os.path.join(annotations_folder, "annotations.json")
    with open(annotations_file, "r") as file:
        data_annotations_file = json.load(file)

    datasets = split_annotations(data_annotations_file)

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    output_annotations_path = output_path / "annotations"
    output_images_path = output_path
    output_images_path.mkdir(parents=True, exist_ok=True)
    output_annotations_path.mkdir(parents=True, exist_ok=True)
    # Save new JSON files
    for split, data in datasets.items():
        with open(os.path.join(output_annotations_path, f"{split}.json"), "w") as f:
            json.dump(data, f, indent=4)
    # Move images

    _ = shutil.move(images_folder, output_images_path)

    shutil.make_archive(
        Path(args.zip) / output_path.name,
        "zip",
        output_path.parent,
        output_path.name,
    )
