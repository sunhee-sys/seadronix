import cv2
import json
import shutil
import argparse
import numpy as np
import scipy.io as sio
from enum import Enum
from pathlib import Path


class SingaporeMaritimeDataset:
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

    def __init__(self, data_path: Path, image_size: tuple = (640, 640)):
        self.data_path = data_path
        self.image_size = image_size
        self.original_image_size = None
        self.videos_ground_truth = self.retrieve_ground_truth()
        self.frames = self.retrieve_frames()
        self.ground_truth = list()
        # This is needed as the labeled videos where longer and cropped before release.
        for video_name in self.frames.keys():
            self.ground_truth.append(
                self.videos_ground_truth[video_name][: len(self.frames[video_name])]
            )
        self.frames = np.concatenate(list(self.frames.values()))
        cv2.destroyAllWindows()
        self.ground_truth = np.concatenate(self.ground_truth)
        self.clean_videos_ground_truth()

    def retrieve_ground_truth(self):
        gt_path = self.data_path / "ObjectGT"
        videos_gt = dict()
        for gt_file in gt_path.glob("*.mat"):
            video_name = "_".join(gt_file.stem.split("_")[:-1])
            mat = sio.loadmat(gt_file)
            videos_gt[video_name] = mat["structXML"][0]
        return videos_gt

    def retrieve_frames(self):
        video_path = self.data_path / "Videos"
        videos_frames = dict()
        for video_name in self.videos_ground_truth.keys():
            video_frames = list()
            video = cv2.VideoCapture(video_path / f"{video_name}.avi")
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                if self.original_image_size is None:
                    self.original_image_size = frame.shape[:2]
                frame = cv2.resize(
                    frame, dsize=self.image_size, interpolation=cv2.INTER_CUBIC
                )
                video_frames.append(frame)
            video.release()
            videos_frames[video_name] = np.stack(video_frames)
        return videos_frames

    def clean_videos_ground_truth(self):
        h_scale = self.image_size[0] / self.original_image_size[0]
        w_scale = self.image_size[1] / self.original_image_size[1]
        for gt in self.ground_truth:
            for bb in gt["BB"]:
                if len(bb) == 0:
                    continue
                bb[0] *= w_scale
                bb[1] *= h_scale
                bb[2] *= w_scale
                bb[3] *= h_scale


class SingaporeMaritimeToCocoFormat:
    def __init__(self, dataset: SingaporeMaritimeDataset, path: Path):
        self.dataset = dataset
        self.path = path

    def save_images(self):
        save_path = self.path / "images"
        save_path.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(self.dataset.frames):
            cv2.imwrite(save_path / f"frame_{i:04}.jpg", frame)

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

        # Fill images
        image_size = self.dataset.frames.shape[1:3]
        content["images"] = [
            {
                "id": int(i),
                "file_name": f"frame_{i:04}.jpg",
                "width": image_size[0],
                "height": image_size[1],
                "tag": "REAL",
                "split": splits[i],
            }
            for i in indexes
        ]

        # Fill annotations
        content["annotations"] = list()
        idx = 0
        for image_id in indexes:
            for i in range(len(self.dataset.ground_truth[image_id]["Object"])):
                if len(self.dataset.ground_truth[image_id]["Object"][i]) == 0:
                    continue
                content["annotations"].append(
                    {
                        "id": idx,
                        "image_id": int(image_id),
                        "category_id": int(
                            self.dataset.ground_truth[image_id]["Object"][i][0]
                        ),
                        "bbox": self.dataset.ground_truth[image_id]["BB"][i].tolist(),
                    }
                )
                idx += 1

        # Fill categories
        content["categories"] = [
            {"id": category.value, "name": category.name}
            for category in self.dataset.Categories
        ]

        with open(path, "w") as file:
            json.dump(content, file, indent=4)

    def generate_coco_annotation_files(
        self,
        version: str,
        train_val_test_ratio: list,
        description: str,
    ) -> None:
        frames_nb = len(self.dataset.frames)

        indexes = np.arange(frames_nb)
        save_path = self.path / "annotations"
        save_path.mkdir(parents=True, exist_ok=True)

        train_ratio, val_ratio, test_ratio = train_val_test_ratio
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
            raise ValueError("Train, validation, and test ratios must sum up to 1.")

        # Shuffle and split indices: 80% Train, 10% Val, 10% Test
        np.random.shuffle(indexes)
        train_split = int(train_ratio * frames_nb)
        val_split = train_split + int(val_ratio * frames_nb)

        # Assign split types
        splits = {}
        splits.update({i: "TRAIN" for i in indexes[:train_split]})
        splits.update({i: "VAL" for i in indexes[train_split:val_split]})
        splits.update({i: "TEST" for i in indexes[val_split:]})

        self.create_coco_annotations_file(
            save_path / "annotations.json", indexes, splits, version, description
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-path", type=str, help="Path to the original Singapore Dataset."
    )
    parser.add_argument(
        "-is",
        "--image-size",
        type=int,
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

    args = parser.parse_args()

    data_path = Path(args.data_path)
    if data_path.suffix == ".zip":
        shutil.unpack_archive(data_path, data_path.parent)
        data_path = data_path.parent / data_path.stem

    dataset = SingaporeMaritimeDataset(
        data_path=data_path, image_size=[args.image_size, args.image_size]
    )

    output_path = Path(args.output_path)

    coco_convertor = SingaporeMaritimeToCocoFormat(dataset, output_path)
    if args.annotations:
        coco_convertor.generate_coco_annotation_files(
            version=args.version,
            train_val_test_ratio=args.train_val_test_ratio,
            description="Coco annotations for singapore maritime dataset",
        )
    if args.images:
        coco_convertor.save_images()
    if args.zip:
        shutil.make_archive(
            Path(args.zip) / output_path.name,
            "zip",
            output_path.parent,
            output_path.name,
        )
