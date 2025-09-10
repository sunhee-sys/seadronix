import random
import logging
import time
import ALSLib.ALSTestManager
from ALSLib.ALSHelperDataCollection import (
    SensorThreadManager,
    SensorOptions,
)

_L = logging.getLogger(__name__)
_L.setLevel(0)


def data_generation_message_handler(raw_message):
    pass


def EnvironmentDemoScenario():
    options = SensorOptions()
    options.out_dir = ".\outputs"

    options.clear_dir = True
    options.burn_in_samples = 10
    options.batch_limit = 512
    options.save_metadata = True

    mgr = SensorThreadManager()
    mgr.auto_setup(TestManager.client, options, "{sensor_number}__{frame_id}")
    mgr.log_enable(True)
    mgr.start()

    mgr.collect_samples_until_paused()
    mgr.stop()

    TestManager.SaveReplayFiles()
    TestManager.SaveSensorDataFilesToS3(options.out_dir)
    TestManager.SetTestStatusSuccess()


if __name__ == "__main__":
    TestManager = ALSLib.ALSTestManager.ALSTestManager(data_generation_message_handler)
    TestManager.StartTest(EnvironmentDemoScenario)