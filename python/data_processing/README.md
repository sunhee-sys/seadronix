# Data Processing Module

> Tools to preprocess, convert, split, and merge maritime datasets into COCO format for deep learning pipelines.

---

## Overview

This module provides a suite of utilities for:

- Converting **AILS** and **Singapore Maritime** datasets to COCO format
- Visualizing bounding boxes and extracting class names
- Merging and splitting COCO-style datasets
- Generating training-ready dataset folders

---

## Scripts and What They Do

### `ails_to_coco_format.py`

Converts an **AILiveSim dataset** into COCO format with custom categories.

**Key Arguments:**

- `--data-path`: Path to the original AILS dataset (zip or folder)
- `--image-size`: Target image resolution
- `--previous-data-path`: Previous dataset for incremental merge
- `--output-path`: Where to save the dataset
- `--annotations`, `--images`, `--zip`: Control dataset generation

**Example:**

```bash
python ails_to_coco_format.py \
  --data-path /path/to/AILS.zip \
  --image-size 640 \
  --previous-data-path /path/to/previous_dataset.zip \
  --output-path /path/to/output \
  --annotations --images \
  --zip /path/to/zip_file
```

---

### `singapore_maritime_to_coco_format.py`

Converts the **Singapore Maritime Dataset** to COCO format.

**Key Arguments:**

- `--data-path`: Path to dataset with `Videos` and `ObjectGT`
- `--output-path`: Output COCO dataset path
- `--image-size`: Resize frames (default: 640x640)
- `--annotations`, `--images`, `--zip`: Control export behavior

**Example:**

```bash
python singapore_maritime_to_coco_format.py \
  --data-path /path/to/singapore_dataset \
  --image-size 640 \
  --output-path /path/to/output \
  --annotations --images \
  --zip /path/to/zip_file

```

---

### `ails_draw_bbox.py`

Visualize and debug bounding boxes on sample images from the AILS dataset.

**Key Arguments:**

- `--data-path`: Path to the dataset

**Example:**

```bash
python ails_draw_bbox.py --data-path /path/to/AILS
```

---

### `ails_list_classes.py`

Extract and print all unique object classes from an AILS-style dataset.

**Key Arguments:**

- `--data-path`: Path to the dataset

**Example:**

```bash
python ails_list_classes.py --data-path /path/to/AILS
```

---

### `merge_coco_datasets.py`

Merges multiple COCO datasets (e.g., AILS + Singapore Maritime).

**Key Arguments:**

- `--data-path`: Comma-separated list of datasets to merge
- `--output-path`: Where to save merged dataset
- `--zip`: Optional zip output

**Example:**

```bash
python merge_coco_datasets.py \
  --data-path /path/to/AILS.zip,/path/to/singapore_dataset.zip \
  --output-path /path/to/merged_output \
  --zip /path/to/zip_file
```

---

### `split_coco.py`

Splits a COCO dataset into:

- Train
- Val
- Test (ID, OOD, REAL)

**Key Arguments:**

- `--data-path`: Path to COCO dataset (zip or folder)
- `--train_val_test_ratio`: Train/val/test split ratio (must sum to 1)
- `--output-path`: Output folder
- `--zip`: Optional zipped output

**Example:**

```bash
python split_coco.py \
  --data-path /path/to/coco_dataset.zip \
  --train_val_test_ratio 0.8,0.1,0.1 \
  --output-path /path/to/split_output \
  --zip /path/to/zip_file
```

---

## Project Structure

```
data_processing/
├── JSON2YOLO
│   ├── LICENSE
│   ├── README.md
│   ├── __pycache__
│   │   └── utils.cpython-310.pyc
│   ├── general_json2yolo.py
│   ├── labelbox_json2yolo.py
│   └── utils.py
├── README.md
├── ails_draw_bbox.py
├── ails_list_classes.py
├── ails_to_coco_format.py
├── merge_coco_datasets.py
├── singapore_maritime_to_coco_format.py
└── split_coco.py

```
---

## JSON2YOLO Directory

This directory is a utility adapted from the official YOLO ecosystem. It provides scripts to convert various JSON annotation formats (e.g., Labelbox, generic formats) into YOLO format.

You don't need to modify these files unless you are adding support for a new annotation tool. For more details, refer to the official YOLO documentation or [Ultralytics GitHub repository](https://github.com/ultralytics/JSON2YOLO).