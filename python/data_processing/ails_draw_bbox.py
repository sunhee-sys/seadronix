import argparse
import json
from pathlib import Path

import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-path", type=str, help="Path to the original Singapore Dataset."
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)

    font = cv2.FONT_HERSHEY_SIMPLEX

    scenario_situation_paths = [p for p in data_path.iterdir() if p.is_dir()]
    labels = list()
    # for scenario_situation_path in scenario_situation_paths:
    #     weathers_path = [p for p in scenario_situation_path.iterdir() if p.is_dir()]
    #     for weather_path in weathers_path:
    #         if weather_path.stem != "5_NoonClear_Beaufort_01":
    #             continue
    #         for annotation_path in weather_path.glob("*.json"):
    for annotation_path in [
        Path(
            "/home/adrien/Repositories/AILiveSim/Fenrir/data/ML_dataset/ML_Ocean_3_Far_Big/5_NoonClear_Beaufort_01/Cameras_CAM0_420.json"
        ),
        Path(
            "/home/adrien/Repositories/AILiveSim/Fenrir/data/ML_dataset/ML_Ocean_3_Far_Big/5_NoonClear_Beaufort_01/Cameras_CAM0_421.json"
        ),
        Path(
            "/home/adrien/Repositories/AILiveSim/Fenrir/data/ML_dataset/ML_Ocean_3_Far_Big/5_NoonClear_Beaufort_01/Cameras_CAM0_422.json"
        ),
    ]:
        frame_path = annotation_path.parent / f"{annotation_path.stem}.png"
        frame = cv2.imread(frame_path)
        with open(annotation_path) as f:
            annotation = json.load(f)
        tracked_objects = annotation["received_metadata"].get("TrackedObj")
        if not tracked_objects:
            continue
        for tracked_object in tracked_objects:
            label = "_".join(tracked_object["ObjectName"].split("_")[:-2])
            cv2.rectangle(
                frame,
                (
                    int(tracked_object["BB2D"][0]["X"]),
                    int(tracked_object["BB2D"][0]["Y"]),
                ),
                (
                    int(tracked_object["BB2D"][1]["X"]),
                    int(tracked_object["BB2D"][1]["Y"]),
                ),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                label,
                (
                    int(tracked_object["BB2D"][0]["X"]),
                    int(tracked_object["BB2D"][0]["Y"] - 10),
                ),
                font,
                0.5,
                (0, 255, 0),
                2,
            )
        cv2.imshow(f"{annotation_path.stem}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
