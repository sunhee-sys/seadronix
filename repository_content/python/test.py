import argparse
import json
import os
from pathlib import Path
import shutil
from ultralytics import YOLO
import yaml


# VH_OUTPUTS_DIR = os.getenv("VH_OUTPUTS_DIR")
# metric_output_dir = os.path.join(VH_OUTPUTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-path", type=str, help="Path to the Singapore Maritime Dataset."
    )
    parser.add_argument(
        "-dc",
        "--data-config-path",
        type=str,
        help="Path to the Singapore Maritime Dataset yaml config.",
    )
    parser.add_argument(
        "-m", "--model-path", type=str, help="Path to the yolo pretrained model."
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Path to the folder where the jsons will be saved.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if data_path.suffix == ".zip":
        shutil.unpack_archive(data_path, data_path.parent)

    model = YOLO(Path(args.model_path))

    # Get images
    data_config_path = Path(args.data_config_path)
    with open(data_config_path, "r") as file:
        data_config = yaml.safe_load(file)
    test_folders = data_config.get("test", "")
    # Looping over all the different test sets: test_real, test_id and test_ood
    for test_folder in test_folders:
        test_folder_name = test_folder.split("/")[-1]
        test_images_path = Path(data_config.get("path", "")) / test_folder
        images_paths = list(test_images_path.glob("*"))
        if images_paths:  # If list is not empty
            results = model(images_paths)

            results = [
                {
                    "image_id": int(Path(r.path).stem.split("_")[-1]),
                    "detections": json.loads(r.to_json()),
                }
                for r in results
            ]
        else:
            print(f"No images found in {test_images_path}. Skipping model inference.")
            results = {"message": "No images were found in this split"}
        result_json_path = Path(args.output_path) / (test_folder_name + "_results.json")
        try:
            with open(Path(result_json_path), "w") as file:
                json.dump(results, file, indent=4)
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error writing to {result_json_path}: {e}")
