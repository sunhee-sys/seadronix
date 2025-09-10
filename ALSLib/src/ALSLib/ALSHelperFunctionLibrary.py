import os, struct, pathlib, time, os.path as ospath
from . import ALSHelperImageLibrary as ALSImg
from importlib.resources import files as lib_files
from enum import Enum


# Some functions that can be used to help overrides
# Basic stream helpers can also be used as reference
def average(numbers):
    if len(numbers) == 0:
        return 0
    total = sum(numbers)
    return total / len(numbers)


def convert_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return hours, minutes, remaining_seconds


def GenerateFilesForConfig(ListOfFiles, ListOfPathLists, ListOfValueLists):
    for i in enumerate(zip(ListOfFiles, ListOfPathLists, ListOfValueLists)):
        BackupFile(i[1][0])
        with open(i[1][0], "r+", newline="") as config:
            config_Lines = config.readlines()
            Lines = ReplaceVariablesWithValues(i[1][1], i[1][2], config_Lines)
            config.seek(0)
            config.writelines(Lines)
    return


def ReadOverridesFromCSV(FilePath):
    with open(FilePath) as csv:
        Lines = csv.readlines()

    file_names = []
    variable_paths = []
    values = []

    for line in Lines:
        line = (
            line.replace("\n", "").replace("\t", "").replace("\r", "").replace(" ", "")
        )
        fields = line.split(";")
        file_names.append(fields[0])
        variable_paths.append(fields[1])
        values.append(fields[2])

    return zip(file_names, variable_paths, values)


def BackupFile(FileToBackup):
    with open(FileToBackup, newline="") as file:
        Lines = file.readlines()

    backup_file = os.path.splitext(FileToBackup)[0] + ".ALSBackup"
    with open(backup_file, "w", newline="") as backup:
        backup.writelines(Lines)

    return


def Restore(FilePath):
    backup_file = os.path.splitext(FilePath)[0] + ".ALSBackup"

    if os.path.isfile(backup_file):
        with open(backup_file, newline="") as backup:
            Lines = backup.readlines()

        with open(FilePath, "w", newline="") as file:
            file.writelines(Lines)

        os.remove(backup_file)

    return


def RestoreFromOverrideFile(ConfigFilePath, OverrideFilePath):
    overrides = ReadOverridesFromCSV(OverrideFilePath)
    overrides = sorted(overrides, key=lambda x: x[0].lower())

    ListOfFiles = []

    ListOfFiles = list(
        set(map(lambda x: os.path.join(ConfigFilePath, x[0]), overrides))
    )

    for file in ListOfFiles:
        Restore(file)


def GenerateConfigsFromOverrides(ConfigFilePath, OverrideFilePath):
    # iterable of 3 tuples with filename, var_path and value
    overrides = ReadOverridesFromCSV(OverrideFilePath)
    overrides = sorted(overrides, key=lambda x: x[0].lower())

    list_of_files = []
    list_of_variable_path_lists = []
    list_of_value_lists = []

    while len(overrides) > 0:
        filename = overrides[0][0]
        overrides_for_file = list(filter(lambda x: x[0] == filename, overrides))
        overrides = list(filter(lambda x: x[0] != filename, overrides))
        list_of_variable_paths = []
        list_of_values = []

        for i in enumerate(overrides_for_file):
            list_of_variable_paths.append(i[1][1])
            list_of_values.append(i[1][2])

        list_of_files.append(os.path.join(ConfigFilePath, filename))
        list_of_variable_path_lists.append(list_of_variable_paths)
        list_of_value_lists.append(list_of_values)

    GenerateFilesForConfig(
        list_of_files, list_of_variable_path_lists, list_of_value_lists
    )


# replaces given path values with given values
def ReplaceVariablesWithValues(ListPaths, ListVars, Lines):
    if len(ListPaths) < 1:
        print("No variable paths provided for generation!")
    if len(ListVars) < 1:
        print("No variable values provided for generation!")

    # get rid of entries without pairs
    sorted_zip = zip(ListPaths, ListVars)
    ListPaths, ListVars = zip(*sorted_zip)
    current_path = ""
    depth = 0
    last_depth = 0

    # Walk trough file
    NewLines = []
    for i in range(len(Lines)):
        line = (
            Lines[i]
            .replace("\n", "")
            .replace("\t", "")
            .replace("\r", "")
            .replace(" ", "")
        )
        new_line = Lines[i]
        # new section
        if line.startswith("[") and line.endswith("]"):
            current_path = line[1:-1]
        # if already in a section
        elif current_path != "":
            if "=" in line:
                var_name = "." + line.split("=")[0]
                parts = current_path.split(".")
                if len(parts) > 1:
                    if depth != last_depth:
                        current_path += var_name
                        last_depth = depth
                    else:
                        current_path = ".".join(parts[0:-1]) + var_name
                else:
                    current_path += var_name
                if line.endswith("{"):
                    depth += 1

                # check against list of paths
                for item in enumerate(ListPaths):
                    path = item[1]
                    variable = ListVars[item[0]]
                    if current_path == path:
                        line_parts = new_line.split("=")
                        new_line = line_parts[0] + "=" + variable + "\r\n"

            if line.endswith("}"):
                depth -= 1
                path = current_path.split(".")
                current_path = ".".join(path[0 : depth + 1])

        NewLines.append(new_line)
    return NewLines


def ConvertFileToParameterFormat(FileText):
    FileText = FileText.replace(" ", "&nbsp")
    FileText = FileText.replace("\r\n", "<br>")
    FileText = FileText.replace("\r", "<br>")
    FileText = FileText.replace("\n", "<br>")
    FileText = FileText.replace("\t", "&t")
    return FileText


def ConvertStringToParameterFormat(FileText):
    return ConvertFileToParameterFormat(FileText)


def format_override_value(override):
    return override.replace("\n", "<p>")


def build_override_path(file_name, path_in_file):
    result = file_name
    if result[-1] != ";":
        result += ";"
    result += path_in_file
    if result[-1] != ";":
        result += ";"
    return result


def combine_overrides(*overrides):
    result = str()
    for override in overrides:
        result += override + "<br>"
    return result.removesuffix("<br>")


def build_override(file_name, path_in_file, value):
    return build_override_path(file_name, path_in_file) + format_override_value(value)


def build_override_array_value(array_entries):
    value = "{\n(\n"
    for i, entry in enumerate(array_entries):
        value += f"[{i}]={entry}\n"
    value += ")\n}\n"
    return format_override_value(value)


def build_override_array(file_name, array_path, array_entries):
    return build_override(
        file_name, array_path, build_override_array_value(array_entries)
    )


def build_override_vector_value(x, y, z, w=None):
    value = "{\n(\n" + f"X={x}\nY={y}\nZ={z}\n"
    if w != None:
        value += f"W={w}\n"
    value += ")\n}\n"
    return format_override_value(value)


def build_override_vector(file_name, vector_path, x, y, z, w=None):
    return build_override(
        file_name, vector_path, build_override_vector_value(x, y, z, w)
    )


def build_override_transform_value(translation, rotation, scale):
    if min(len(rotation), len(translation), len(scale)) < 3:
        print("Error: invalid vector in transform override. >= 3 components required.")
    value = "{\n(\n"
    if len(rotation) > 3:
        value += "Rotation=" + build_override_vector_value(
            rotation[0], rotation[1], rotation[2], rotation[3]
        )
        value += "Translation=" + build_override_vector_value(
            translation[0], translation[1], translation[2]
        )
        value += "Scale3D=" + build_override_vector_value(scale[0], scale[1], scale[2])
    elif len(rotation) == 3:
        # Euler values only (PositionEuler)
        value += "Location=" + build_override_vector_value(
            translation[0], translation[1], translation[2]
        )
        value += "Rotation=" + build_override_vector_value(
            rotation[0], rotation[1], rotation[2]
        )
        value += "Scale=" + build_override_vector_value(scale[0], scale[1], scale[2])
    value += ")\n}\n"
    return format_override_value(value)


def build_override_transform(file_name, vector_path, location, rotation, scale):
    return build_override(
        file_name,
        vector_path,
        build_override_transform_value(location, rotation, scale),
    )


def overrides_to_dict(overrides_str: str, strip_filename: bool = False):
    overrides_list = list(filter(None, overrides_str.split("<br>")))
    overrides_dict = {}
    for override in overrides_list:
        file, key, value = override.split(";")
        key = key if strip_filename else f"{file};{key}"
        overrides_dict[key] = value
    return overrides_dict


def ReadUint32(buffer, index):
    return (int(struct.unpack("<L", buffer[index : index + 4])[0]), index + 4)


def ReadUint8(buffer, index):
    return (
        int.from_bytes(
            struct.unpack("<c", buffer[index : index + 1])[0],
            byteorder="big",
            signed=False,
        ),
        index + 1,
    )


def ReadString(buffer, index):
    str_val_len, index = ReadUint32(buffer, index)
    str_val = buffer[index : index + str_val_len]
    str_val = str_val.decode("utf-8")
    return (str_val, index + str_val_len)


class SensorImageFormat(Enum):
    Raw = 0
    PNG = 1


def ReadImage_Stream(buffer, index):
    format_int, index = ReadUint8(buffer, index)
    image_format = SensorImageFormat(format_int)
    if image_format == SensorImageFormat.PNG:
        image_size, index = ReadUint32(buffer, index)
    else:  # raw
        image_height, index = ReadUint32(buffer, index)
        image_width, index = ReadUint32(buffer, index)
        image_channels, index = ReadUint32(buffer, index)

    if image_format == SensorImageFormat.PNG:
        # NOTE: here we get directly the png data
        # it can be stored directly in a .png file
        # if you want to further process the image, you will need to
        # unpack it first
        img = buffer[index:image_size]
        image_width = 0
        image_height = 0
        image_channels = 4
        index += image_size

    else:
        image_size = image_height * image_width * image_channels
        image_data = buffer[index : index + image_size]
        index += image_size
        img = ALSImg.DecodeImageData(
            image_data, image_width, image_height, image_channels
        )
    return img, index, image_width, image_height, image_format


def ReadImage_Group(buffer, index):
    image_width, index = ReadUint32(buffer, index)
    image_height, index = ReadUint32(buffer, index)
    image_channels, index = ReadUint8(buffer, index)
    format_int, index = ReadUint8(buffer, index)
    image_format = SensorImageFormat(format_int)
    image = None
    if image_format == SensorImageFormat.PNG:
        # NOTE: here we get directly the png data
        # it can be stored directly in a .png file
        # if you want to further process the image, you will need to
        # unpackit first
        compressed_size, index = ReadUint32(buffer, index)
        image = buffer[index:compressed_size]
    else:
        image_size = image_width * image_height * image_channels
        image_data = buffer[index : index + image_size]
        index += image_size
        image = ALSImg.DecodeImageData(
            image_data, image_width, image_height, image_channels
        )
    return image, index, image_width, image_height, image_format


def find_install_path(require_valid_path: bool = True):
    cd = os.path.abspath("./")
    install_path = None
    while (not install_path) and (os.path.split(cd)[1] != ""):
        content = [os.path.join(cd, f) for f in os.listdir(cd)]
        folders = [os.path.split(f)[1] for f in content if os.path.isdir(f)]
        files = [os.path.split(f)[1] for f in content if os.path.isfile(f)]
        # The main application folder is expected to have one of these files.
        # Do this instead of checking for the ConfigFiles folder, as that may exist elsewhere. -SL
        if ("AILiveSim.exe" in files) or ("AILiveSim.uproject" in files):
            install_path = cd
        else:
            cd = os.path.split(cd)[0]

    if not install_path:
        install_path = os.environ.get("AILIVESIM_PATH")

    if require_valid_path and not install_path:
        raise FileNotFoundError(
            "Could not find the installation path. Is the script running in the correct directory?"
        )

    return install_path


def get_root_folder_path(folder_name: str, add_on_path: str = None):
    final_path = os.path.join(find_install_path(require_valid_path=True), folder_name)
    if add_on_path:
        final_path = os.path.normpath(
            os.path.join(final_path, add_on_path.strip("/").strip("\\"))
        )
    return final_path


def count_directories(root_dir: str):
    count = 0
    with os.scandir(root_dir) as iterator:
        for entry in iterator:
            if entry.is_dir():
                count += 1
    return count


def count_directories_with_substring(root_dir: str, substring: str):
    count = 0
    with os.scandir(root_dir) as iterator:
        for entry in iterator:
            if entry.is_dir() and substring in entry.path:
                count += 1
    return count


def get_directories_with_substring(root_dir: str, substring: str):
    directories = []
    with os.scandir(root_dir) as iterator:
        for entry in iterator:
            if entry.is_dir() and substring in entry.path:
                directories.append(entry.path)
    return directories


def get_directories(root_dir: str):
    directories = []
    with os.scandir(root_dir) as iterator:
        for entry in iterator:
            if entry.is_dir():
                directories.append(entry.path)
    return directories


def get_sensordata_path(add_on_path: str = None):
    return get_root_folder_path("SensorData", add_on_path)


def get_config_path(add_on_path: str = None):
    return get_root_folder_path("ConfigFiles", add_on_path)


def get_lib_resource(resource_file_name: str, binary_mode: bool = False):
    traversable = lib_files("ALSLib").joinpath(resource_file_name)
    if not binary_mode:
        return traversable.read_text()
    else:
        return traversable.read_bytes()


def find_unique_directory_name(base: str):
    index = 0
    directory = f"{base}"

    while ospath.isdir(get_sensordata_path(directory)):
        index += 1
        directory = f"{base}_{index}"
    return directory


def save_override_info(info: str, name: str, dirname: str):
    '''columns = str()
    values = str()
    info = info.split("<br>")
    for x in info:
            override = x.split(";")
            if len(override) < 2:
                    continue

            values += f"{override[-1]}, "
            columns += f"{override[-2]}, "

    info = f"{columns}\n{values.rstrip(', ')}"'''

    out_dir_path = get_sensordata_path(dirname)
    file_path = ospath.join(out_dir_path, f"{name}.txt")

    while not pathlib.Path.exists(pathlib.Path(out_dir_path)):
        time.sleep(0.05)

    with open(file_path, "x", encoding="utf-8") as file:
        file.write(info.replace("<br>", "\n\r"))
