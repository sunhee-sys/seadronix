import argparse
import os
from pathlib import Path
import shutil
from ultralytics import YOLO


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
        "-e",
        "--epochs",
        type=int,
        help="Train number of epochs.",
    )
    parser.add_argument(
        "-is",
        "--image-size",
        type=int,
        help="Image size of the dataset images.",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default=None,
        help="Exported model format.",
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
    model.train(
        data=Path(args.data_config_path),
        epochs=args.epochs,
        imgsz=args.image_size,
        project=output_path,  # metric_output_dir,
    )

    model.val()  # evaluate model performance on the validation set

    shutil.copy(output_path / "train/weights/best.pt", output_path)

    if args.format:
        path = model.export(format=args.format)

    # # Copy the exported model to the Valohai outputs directory
    # shutil.copy(path, '/valohai/outputs/')

    # file_metadata = {
    #     "valohai.alias": "latest-model"
    # }

    # Attach the metadata to the file, enabling easy retrieval by the alias
    # with open("/valohai/outputs/best.onnx.metadata.json", "w") as outfile:
    #     outfile.write(json.dumps(file_metadata))
