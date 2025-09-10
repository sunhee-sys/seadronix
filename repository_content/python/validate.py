import argparse
import os
from pathlib import Path
import shutil

import yaml
from ultralytics import YOLO

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
        help="Path where the coco version of the dataset will be saved.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if data_path.suffix == ".zip":
        shutil.unpack_archive(data_path, data_path.parent)
        data_path = data_path.parent / data_path.stem

    output_path = Path(args.output_path)
    model = YOLO(Path(args.model_path))

    output_path = Path(args.output_path)
    res = model.val(data=Path(args.data_config_path), split="test")
