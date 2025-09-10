import argparse
import json
from pathlib import Path
import shutil

from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-path",
        type=lambda arg: arg.split(","),
        help="Path to the original COCO Datasets.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Path where the coco version of the dataset will be saved.",
    )
    parser.add_argument(
        "-z",
        "--zip",
        type=str,
        default=None,
        help="Zip path, if None, don't zip the result folder",
    )
    parser.add_argument(
        "-v", "--version", default="0.1.0", type=str, help="Version of the annotations."
    )
    parser.add_argument(
        "-de",
        "--description",
        default="Merge of COCO datasets",
        type=str,
        help="Description of the annotations.",
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_images_path = output_path / "images"
    output_images_path.mkdir(parents=True, exist_ok=True)
    annotations_output_path = output_path / "annotations"
    annotations_output_path.mkdir(parents=True, exist_ok=True)

    image_index = 0
    annotations = dict()
    for data_path in args.data_path:
        data_path = Path(data_path)
        if data_path.suffix == ".zip":
            shutil.unpack_archive(data_path, data_path.parent)
        data_path = data_path.parent / data_path.stem
        images_path = data_path / "images"
        frames_path = images_path.glob("*")
        indexes = dict()
        for frame_path in tqdm(frames_path, desc=f"Merging {data_path.name}"):
            shutil.copy(
                frame_path,
                output_path
                / "images"
                / f"{frame_path.stem}_{image_index:05}{frame_path.suffix}",
            )
            indexes[int(frame_path.stem.split("_")[-1])] = image_index
            image_index += 1

        annotations_path = data_path / "annotations"
        annotations_files = annotations_path.glob("*.json")
        for annotation_file in annotations_files:
            annotation_file_name = annotation_file.stem
            with open(annotation_file) as f:
                annotation = json.load(f)
            if annotation_file_name not in annotations:
                annotations[annotation_file_name] = {
                    "info": {"version": args.version, "description": args.description},
                    "images": list(),
                    "annotations": list(),
                    "categories": annotation.get("categories", {}),
                    "annotation_index": 0,
                }
            for ann_image in annotation["images"]:
                new_index = indexes[ann_image["id"]]
                ann_image["id"] = new_index
                splitted_file_name = ann_image["file_name"].split(".")
                ann_image["file_name"] = (
                    ".".join(splitted_file_name[:-1])
                    + f"_{new_index:05}."
                    + splitted_file_name[-1]
                )
                ann_image["tag"] = ann_image.get("tag", "UNKNOWN")
                ann_image["split"] = ann_image.get("split", "UNKNOWN")
            annotations[annotation_file_name]["images"] += annotation["images"]
            for ann_annotation in annotation["annotations"]:
                ann_annotation["id"] = annotations[annotation_file_name][
                    "annotation_index"
                ]
                ann_annotation["image_id"] = indexes[ann_annotation["image_id"]]
                annotations[annotation_file_name]["annotation_index"] += 1
            annotations[annotation_file_name]["annotations"] += annotation[
                "annotations"
            ]

    for annotation_file_name, annotation_content in annotations.items():
        del annotation_content["annotation_index"]
        with open(
            annotations_output_path / f"{annotation_file_name}.json", "w"
        ) as file:
            json.dump(annotation_content, file, indent=4)

    if args.zip:
        shutil.make_archive(
            Path(args.zip) / output_path.name,
            "zip",
            output_path.parent,
            output_path.name,
        )
