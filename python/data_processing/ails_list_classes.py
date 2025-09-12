import argparse
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-path", type=str, help="Path to the original Singapore Dataset."
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)

    scenario_situation_paths = [p for p in data_path.iterdir() if p.is_dir()]
    labels = list()
    for scenario_situation_path in scenario_situation_paths:
        weathers_path = [p for p in scenario_situation_path.iterdir() if p.is_dir()]
        for weather_path in weathers_path:
            for annotation_path in weather_path.glob("*.json"):
                with open(annotation_path) as f:
                    annotation = json.load(f)
                tracked_objects = annotation["received_metadata"].get("TrackedObj")
                if not tracked_objects:
                    continue
                for tracked_object in tracked_objects:
                    label = "_".join(tracked_object["ObjectName"].split("_")[:-2])
                    if label not in labels:
                        labels.append(label)
    print(labels)
