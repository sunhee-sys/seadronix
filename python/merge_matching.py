import argparse
import json
from pathlib import Path
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-path", type=str, help="Path to COCO format dataset."
    )
    parser.add_argument(
        "-sm",
        "--scenario-matching",
        type=str,
        help="Path to the scenario matching results.",
    )
    parser.add_argument(
        "-wm",
        "--weather-matching",
        type=str,
        help="Path to the weather matching results.",
    )
    parser.add_argument(
        "-ssm",
        "--sea-state-matching",
        type=str,
        help="Path to the sea state matching results.",
    )
    parser.add_argument(
        "-z",
        "--zip",
        type=str,
        default=None,
        help="Zip path, if None, don't zip the result folder",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if data_path.suffix == ".zip":
        shutil.unpack_archive(data_path, data_path.parent)
        data_path = data_path.parent / data_path.stem

    annotations_path = data_path / "annotations" / "annotations.json"
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    with open(Path(args.scenario_matching), "r") as f:
        scenario_matching_results = json.load(f)

    with open(Path(args.weather_matching), "r") as f:
        weather_matching_results = json.load(f)

    with open(Path(args.sea_state_matching), "r") as f:
        sea_state_matching_results = json.load(f)

    images_annotations = annotations.get("images", [])
    for annotation in images_annotations:
        image_id = annotation.get("id")
        scenario = scenario_matching_results.get(str(image_id))
        weather = weather_matching_results.get(str(image_id))
        sea_state = sea_state_matching_results.get(str(image_id))
        annotation["scenario"] = scenario.get("config_file_key") if scenario else None
        annotation["weather"] = weather.get("config_file_key") if weather else None
        annotation["sea_state"] = (
            sea_state.get("config_file_key") if sea_state else None
        )

    with open(annotations_path, "w") as f:
        json.dump(annotations, f, indent=4)

    if args.zip:
        shutil.make_archive(
            Path(args.zip) / data_path.name,
            "zip",
            data_path.parent,
            data_path.name,
        )
