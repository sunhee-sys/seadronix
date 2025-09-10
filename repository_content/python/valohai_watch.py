import os
import valohai
import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

import csv
import json

VH_OUTPUTS_DIR = os.getenv("VH_OUTPUTS_DIR")
metric_output_dir = os.path.join(VH_OUTPUTS_DIR)


class ValohaiHandler(PatternMatchingEventHandler):
    def on_modified(self, event):
        if ".csv" in event.src_path:
            with open(event.src_path, "r") as file:
                data = list(csv.reader(file, delimiter=","))

                keys = data[0]
                latest_values = data[-1]

                metadata = {}

                for i in range(len(keys)):
                    key = keys[i].strip()
                    value = latest_values[i].strip()

                    metadata[key] = value

                print(json.dumps(metadata))

    def on_created(self, event):
        if (".pt" in event.src_path) or (".onnx" in event.src_path):
            with open(event.src_path, "r"):
                # Choose alias name
                if "best" in event.src_path:
                    alias = "current-best-model"
                    if "onnx" in event.src_path:
                        alias = "production-model"
                else:
                    return

                metadata = {
                    "valohai.alias": alias  # creates or updates a Valohai data alias to point to this output file
                }
                model_name = os.path.basename(event.src_path)
                model_path = os.path.dirname(event.src_path)

                # Save metadata file in the save location as the model
                metadata_path = valohai.outputs().path(
                    os.path.join(model_path, f"{model_name}.metadata.json")
                )
                with open(metadata_path, "w") as outfile:
                    json.dump(metadata, outfile)


if __name__ == "__main__":
    event_handler = ValohaiHandler(patterns=["*.csv", "*.pt", "*.onnx"])
    observer = Observer()
    observer.schedule(event_handler, path=metric_output_dir, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
