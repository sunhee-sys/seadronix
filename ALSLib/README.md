# ALSLib

## How to install
The library and it's dependencies can be installed with a global install as follows.

	python -m pip install .\ALSLib

Should you wish to do changes in the library, and want the changes to be reflected while you are working, please use `--editable`

    python -m pip install --editable .\ALSLib

NOTE: Install path is declared as relative here, and it should reflect your own folder structure.

If you do not want to install package globally, you can create a virtual environment using `venv`

    python -m venv path/to/my_virtual_environment

Activate your new virtual environment by running:

    path/to/my_virtual_environment/Scripts/Activate

Now you can install the package to your virtual environment with `python -m pip install`.

## How to use
```python
    #To import a module from library
    import ALSLib.ALSClient
    #Or to import a class
    from ALSLib.ALSHelperDataCollection import SensorThreadManager, SensorOptions
```

## Features
Library includes following modules:
* ALSClient
* ALSHelperDataCollection
* ALSHelperFunctionLibrary
* ALSHelperLidarLibrary
* ALSPointCloudVisualisation
* ALSRadarVis
* ALSTestManager
* TCPClient
* UDPClient

Library also includes two dynamic link libraries that have been precompiled for win_amd64.
* ALS_TestDefinitionGenerator
* ALSReceiver

## Support
Should you encounter a bug when using the library, please contact support@ailivesim.com

## License
Please refer [LICENSE](/LICENSE) file.
