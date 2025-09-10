import cv2
import json
import shutil
import argparse
import numpy as np
from enum import Enum
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
import copy


def get_xml_node_content_by_name(xml_root: ET.Element, node_name: str) -> ET.Element:
    node = xml_root.find(node_name)  # Replace with actual tag
    if node is None:
        raise ValueError(f"Node {node_name} not found.")
    return node.text


class AILSToCocoFormat:
    class Categories(Enum):
        FERRY = 1
        BUOY = 2
        VESSEL_SHIP = 3
        SPEED_BOAT = 4
        BOAT = 5
        KAYAK = 6
        SAIL_BOAT = 7
        SWIMMING_PERSON = 8
        FLYING_BIRD_PLANE = 9
        OTHER = 10

    class AILSCategories(Enum):
        BP_Buoy_Cone = 2
        BP_CargoBoat2 = 3
        BP_CargoBoat = 3
        BP_BargeBoat_2_Hopper_Cargo = 3
        BP_BargeBoat_1_Hopper_Cargo = 3
        BP_BargeBoat_2 = 3
        BP_BargeBoat_1_BulkCarrier = 3
        OilTanker = 3
        BP_PanamaxOil = 3
        BP_ContainerShip_007 = 3
        BP_ContainerShip_005 = 3
        BP_BargeBoat_2_BulkCarrier = 3
        BP_CargoShipHatches = 3
        BP_ContainerShip_004 = 3
        BP_ContainerShip_003 = 3
        BP_BargeBoat_1 = 3
        BP_ContainerShip_002 = 3
        BP_CargoShipGeneral = 3
        BP_TankerShipShallow = 3
        BP_ContainerShip_001 = 3
        BP_SuperTanker = 3
        BP_ContainerShip_006 = 3
        BP_CargoShipHatchesOpen = 3
        BP_ContainerShip_008 = 3
        BP_FishingBoat_NewOcean = 5
        BP_SailBoat02 = 5
        BP_SailBoat = 5
        BP_MotorBoat01 = 5
        BP_Boat_Zodiac_Rescue_001 = 5
        BP_Boat_Speedboat_Common_003 = 5
        BP_Boat_Zodiac_Patrol_003 = 5
        BP_CruizeShip_NewOcean = 3
        BP_SmallMotorBoat_NO = 3
        BP_Ferry_NewOcean = 1

    def __init__(
        self,
        data_path: Path,
        output_path: Path,
        image_size: tuple = (640, 640),
        previous_data_path: Path = None,
    ):
        self.data_path = data_path
        self.output_path = output_path
        self.image_size = image_size

        self.frames_information = dict()
        self.annotations = []
        self.prev_filenames = set()

        if previous_data_path and previous_data_path.exists():
            print("Previous Data Exists..")
            with open(previous_data_path, "r") as file:
                prev_data = json.load(file)

            # Store previous file names
            self.prev_filenames = set(
                "_".join(Path(f["file_name"]).stem.split("_")[:-1])
                for f in prev_data.get("images", [])
            )

            # Load previous images
            for frame in prev_data.get("images", []):
                self.frames_information[frame["id"]] = frame

            # Load previous annotations
            self.annotations = prev_data.get("annotations", [])

    def save_images(self):
        images_path = self.output_path / "images"
        images_path.mkdir(parents=True, exist_ok=True)

        test_suites_paths = [p for p in self.data_path.iterdir() if p.is_dir()]

        index = max(self.frames_information.keys(), default=-1) + 1

        for test_suite_path in tqdm(test_suites_paths, desc="Test suites", position=0):
            tests_paths = [
                p
                for p in test_suite_path.iterdir()
                if p.is_dir() and p.name != "InstanceLogs"
            ]
            for test_path in tqdm(tests_paths, desc="Tests", position=1):
                # Load replay files to get information about the generation
                replay_file = list(test_path.glob("*.xml"))
                if len(replay_file) < 1:
                    print(f"No replay files has been found for test {test_path.stem}")
                    continue
                replay_file = replay_file[0]
                # Add scenario name
                replay_xml = ET.parse(replay_file)
                replay_root = replay_xml.getroot()
                scenario_name = get_xml_node_content_by_name(
                    replay_root, "ScenarioName"
                )
                weather_name = get_xml_node_content_by_name(replay_root, "Weather")
                sea_state_name = get_xml_node_content_by_name(replay_root, "SeaState")

                for frame_path in tqdm(
                    test_path.glob("*.png"), desc="Images", leave=False
                ):
                    file_name = frame_path.stem

                    if file_name in self.prev_filenames:
                        print(f"[SKIP] {file_name} already in previous dataset.")
                        continue

                    frame = cv2.imread(frame_path)
                    height, width, _ = frame.shape

                    file_name = f"{frame_path.stem}_{index:05}.png"
                    self.frames_information[index] = {
                        "id": int(index),
                        "file_name": file_name,
                        "scenario": scenario_name,
                        "weather": weather_name,
                        "sea_state": sea_state_name,
                        "path": frame_path,
                        "width": self.image_size[1],
                        "height": self.image_size[0],
                        "original_width": width,
                        "original_height": height,
                    }
                    frame = cv2.resize(
                        frame, dsize=self.image_size, interpolation=cv2.INTER_CUBIC
                    )
                    cv2.imwrite(images_path / file_name, frame)
                    index += 1

    @staticmethod
    def resize_bbox(bbox: list, image_size: tuple, original_image_size: tuple) -> list:
        h_scale = image_size[0] / original_image_size[0]
        w_scale = image_size[1] / original_image_size[1]

        x = bbox[0] * w_scale
        y = bbox[1] * h_scale
        width = bbox[2] * w_scale
        height = bbox[3] * h_scale

        return [x, y, width, height]

    def create_coco_annotations_file(
        self,
        path: Path,
        indexes: np.ndarray,
        splits: dict,
        version: str,
        description: str,
    ) -> None:
        content = dict()
        # Generate information
        content["info"] = {
            "version": version,
            "description": description,
        }

        frames = [
            {
                "id": frame_index,
                "file_name": frame_info["file_name"],
                "width": frame_info["width"],
                "height": frame_info["height"],
                "scenario": frame_info["scenario"],
                "weather": frame_info["weather"],
                "sea_state": frame_info["sea_state"],
                "tag": frame_info.get(
                    "tag", "ID"
                ),  # Preserve if present and TO-DO: Add logic for OOD as well...
                "split": frame_info.get(
                    "split", splits[frame_index]
                ),  # Preserve if present
            }
            for frame_index, frame_info in self.frames_information.items()
            if frame_index in indexes
        ]
        content["images"] = frames

        category_list = []

        # Fill annotations
        content["annotations"] = self.annotations.copy()
        idx = max([ann["id"] for ann in self.annotations], default=-1) + 1

        for frame_id in tqdm(indexes, desc="Annotation frame"):
            frame_info = self.frames_information[frame_id]
            if "path" not in frame_info:
                print(
                    f"Skipping frame {frame_id} (no path found, likely from previous dataset)"
                )
                continue
            annotation_path = (
                frame_info["path"].parent / f"{frame_info['path'].stem}.json"
            )
            with open(annotation_path) as f:
                annotation = json.load(f)
            tracked_objects = annotation["received_metadata"].get("TrackedObj")
            if not tracked_objects:
                continue
            for tracked_object in tracked_objects:
                label = "_".join(tracked_object["ObjectName"].split("_")[:-2])
                x1 = tracked_object["BB2D"][0]["X"]
                y1 = tracked_object["BB2D"][0]["Y"]
                x2 = tracked_object["BB2D"][1]["X"]
                y2 = tracked_object["BB2D"][1]["Y"]

                x_min = min(x1, x2)
                y_min = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)

                bbox = self.resize_bbox(
                    [x_min, y_min, width, height],
                    (frame_info["height"], frame_info["width"]),
                    (frame_info["original_height"], frame_info["original_width"]),
                )

                if label in self.AILSCategories.__members__:
                    category_id = int(self.AILSCategories[label].value)
                else:
                    category_id = 10
                    category_list.append(label)

                content["annotations"].append(
                    {
                        "id": idx,
                        "image_id": int(frame_id),
                        "category_id": category_id,
                        "bbox": bbox,
                    }
                )
                idx += 1
        print(set(category_list))
        # Fill categories
        content["categories"] = [
            {"id": category.value, "name": category.name}
            for category in self.Categories
        ]

        with open(path, "w") as file:
            json.dump(content, file, indent=4)

    def generate_coco_annotation_files(
        self,
        version: str,
        train_val_test_ratio: list,
        description: str,
    ) -> None:
        frames_nb = len(self.frames_information.keys())
        original_indexes = np.arange(frames_nb)  # Original indexes

        # Make a deep copy to avoid altering the original order
        shuffled_indexes = copy.deepcopy(original_indexes)
        np.random.shuffle(shuffled_indexes)

        # Paths
        annotations_path = self.output_path / "annotations"
        annotations_path.mkdir(parents=True, exist_ok=True)

        train_ratio, val_ratio, test_ratio = train_val_test_ratio
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
            raise ValueError("Train, validation, and test ratios must sum up to 1.")

        # Shuffle and split indices based on the ratios
        train_split = int(train_ratio * frames_nb)
        val_split = train_split + int(val_ratio * frames_nb)

        # Assign split types
        splits = {}
        splits.update({i: "TRAIN" for i in shuffled_indexes[:train_split]})
        splits.update({i: "VAL" for i in shuffled_indexes[train_split:val_split]})
        splits.update({i: "TEST" for i in shuffled_indexes[val_split:]})

        # Pass the original, unshuffled indexes
        self.create_coco_annotations_file(
            annotations_path / "annotations.json",
            original_indexes,
            splits,
            version,
            description,
        )


if __name__ == "__main__":
    print("Using the new settings")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-path", type=str, help="Path to the original AILS Dataset."
    )
    parser.add_argument(
        "-is",
        "--image-size",
        type=int,
        default=640,
        help="Target Width and Height of the dataset images.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Path where the coco version of the dataset will be saved.",
    )
    parser.add_argument("-v", "--version", type=str, help="Version of the annotations.")
    parser.add_argument(
        "-a",
        "--annotations",
        action="store_true",
        help="If true, generate and save the annotations.",
    )
    parser.add_argument(
        "-i", "--images", action="store_true", help="If true, save the images."
    )
    parser.add_argument(
        "-z",
        "--zip",
        type=str,
        default=None,
        help="Zip path, if None, don't zip the result folder",
    )
    parser.add_argument(
        "-tvr",
        "--train_val_test_ratio",
        type=lambda arg: [float(x) for x in arg.split(",")],
        default=[0.8, 0.1, 0.1],
        help="Train, validation, and test split ratios, separated by commas (e.g., 0.8,0.1,0.1).",
    )
    parser.add_argument(
        "-pd",
        "--previous-data-path",
        type=str,
        default="",
        help="Path to previous zipped ALS dataset",
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)
    if data_path.suffix == ".zip":
        shutil.unpack_archive(data_path, data_path.parent)
        data_path = data_path.parent / data_path.stem

    prev_data_path = None
    tmp_prev_dir = None

    if not (args.previous_data_path == ""):
        prev_data_path = Path(args.previous_data_path)
        if prev_data_path.suffix == ".zip" and prev_data_path.exists():
            # Unpack the zip to a temp folder
            tmp_prev_dir = Path(".tmp_prev_data")
            if tmp_prev_dir.exists():
                shutil.rmtree(tmp_prev_dir)
            shutil.unpack_archive(prev_data_path, tmp_prev_dir)
            prev_data_path = (
                tmp_prev_dir / "coco_dataset" / "annotations" / "annotations.json"
            )

            # Copy previous images
            prev_images_dir = tmp_prev_dir / "coco_dataset" / "images"
            merged_images_dir = Path(args.output_path) / "images"
            merged_images_dir.mkdir(parents=True, exist_ok=True)

            print("Previous image directory:")
            print(prev_images_dir)

            for img_file in prev_images_dir.glob("*.png"):
                shutil.copy(img_file, merged_images_dir / img_file.name)

    output_path = Path(args.output_path)

    dataset = AILSToCocoFormat(
        data_path=data_path,
        output_path=output_path,
        previous_data_path=prev_data_path,
    )

    if args.images:
        dataset.save_images()

    if args.annotations:
        dataset.generate_coco_annotation_files(
            version=args.version,
            train_val_test_ratio=args.train_val_test_ratio,
            description="Coco annotations for AILS dataset",
        )

    if args.zip:
        shutil.make_archive(
            Path(args.zip) / output_path.name,
            "zip",
            output_path.parent,
            output_path.name,
        )
    # Cleaning up
    if tmp_prev_dir and tmp_prev_dir.exists():
        shutil.rmtree(tmp_prev_dir)
