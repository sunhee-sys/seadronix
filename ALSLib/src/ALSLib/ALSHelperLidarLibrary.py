import cv2, math
import threading, struct, math
import numpy as np
import os
from .ALSHelperFunctionLibrary import find_install_path, get_sensordata_path


def get_lidar_header_angle_distance(data):
    sizeofFloat, sizeofInt = 4, 4
    index = 8 * sizeofFloat + 4 * sizeofInt
    (
        posX,
        posY,
        posZ,
        quatW,
        quatX,
        quatY,
        quatZ,
        num_cols,
        num_beams_per_col,
        col_time,
        FrameID,
        ColId,
    ) = struct.unpack("<fffffffiifii", data[0:index])
    pointcloud_data = data[index:]
    return (
        posX,
        posY,
        posZ,
        quatW,
        quatX,
        quatY,
        quatZ,
        num_cols,
        num_beams_per_col,
        col_time,
        FrameID,
        ColId,
        pointcloud_data,
    )


def get_lidar_header(data):
    sizeofFloat = 4
    index = 11
    (
        posX,
        posY,
        posZ,
        quatW,
        quatX,
        quatY,
        quatZ,
        numPoints,
        timeStart,
        timeEnd,
        numberOfBeams,
    ) = struct.unpack("<fffffffffff", data[0 : index * sizeofFloat])
    pointCloudData = data[index * sizeofFloat :]
    return (
        posX,
        posY,
        posZ,
        quatW,
        quatX,
        quatY,
        quatZ,
        numPoints,
        timeStart,
        timeEnd,
        numberOfBeams,
        pointCloudData,
    )


def get_lidar_header_from_group(data, index=0):
    sizeofFloat = 4
    LidarHeaderSize = 11
    (
        posX,
        posY,
        posZ,
        quatW,
        quatX,
        quatY,
        quatZ,
        numPoints,
        timeStart,
        timeEnd,
        numberOfBeams,
    ) = struct.unpack(
        "<fffffffffff", data[index : index + (LidarHeaderSize * sizeofFloat)]
    )
    pointCloudData = data[index * sizeofFloat :]
    index += LidarHeaderSize * sizeofFloat
    return (
        posX,
        posY,
        posZ,
        quatW,
        quatX,
        quatY,
        quatZ,
        numPoints,
        timeStart,
        timeEnd,
        numberOfBeams,
        pointCloudData,
        index,
    )


def get_lidar_header_livox(data, size_float):
    index = 9
    posX, posY, posZ, quatW, quatX, quatY, quatZ, timeStart, numPoints = struct.unpack(
        "<fffffffff", data[0 : index * size_float]
    )
    pointCloudData = data[index * size_float :]
    return (
        posX,
        posY,
        posZ,
        quatW,
        quatX,
        quatY,
        quatZ,
        timeStart,
        numPoints,
        pointCloudData,
    )


def read_livox_data(data, size_uint8, size_float, num_points):
    readPoints = 0
    pointSize = size_float * 5 + (2 * size_uint8)
    livoxPoints = []
    while readPoints < num_points:
        start = readPoints * pointSize
        end = (readPoints + 1) * pointSize
        offsetTime, pointX, pointY, pointZ, intensity, tag, number = struct.unpack(
            "fffffBB", data[start:end]
        )  # ReadLivoxPoint()
        livoxPoints.append((offsetTime, pointX, pointY, pointZ, intensity, tag, number))
        readPoints += 1
    return livoxPoints


def SerializeToPCLFileContent(
    numPoints, posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array
):
    pclFileContent = (
        "# .PCD v.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z rgb\n\
	SIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT %f %f %f %f %f %f %f \n\
	POINTS %d \nDATA ascii\n"
        % (int(numPoints), posX, posY, posZ, quatW, quatX, quatY, quatZ, int(numPoints))
    )
    for p in point_array:
        intensity = 1000
        if not math.isinf(p[3]) and not math.isnan(p[3]) and p[3] is not None:
            intensity = int(p[3])
        pclFileContent += "%.5f %.5f %.5f %d\n" % (p[0], p[1], p[2], intensity)
    return pclFileContent


def create_sensor_data_folders():
    basepath = find_install_path()
    sensordata_path = get_sensordata_path()
    files = os.listdir(basepath)
    has_data = False
    has_pcl = False
    for file in files:
        if os.path.isdir(os.path.join(basepath, file)) and file == "SensorData":
            inner_files = os.listdir(sensordata_path)
            has_data = True
            for inner_file in inner_files:
                if (
                    os.path.isdir(os.path.join(sensordata_path, inner_file))
                    and inner_file == "pcl"
                ):
                    has_pcl = True
    if not has_data:
        os.mkdir(sensordata_path)
    if not has_pcl:
        d_path = os.path.join(sensordata_path, "pcl")
        os.mkdir(d_path)


class convert_distance_to_3d:
    def __init__(self):
        self.cache_theta_sin = []
        self.cache_theta_cos = []

    def cache_theta(self, theta):
        self.cache_theta_sin.append(math.sin(theta))
        self.cache_theta_cos.append(math.cos(theta))

    def cache_phi(self, phi):
        self.cache_phi_cos = math.cos(phi)
        self.cache_phi_sin = math.sin(phi)

    def clear_cached(self):
        self.cache_theta_sin = []
        self.cache_theta_cos = []
        self.cache_phi_cos = 0
        self.cache_phi_sin = 0

    def distance_to_3d(self, readings_array, num_beams_per_col):
        r_array = readings_array[0:num_beams_per_col]
        intensity_array = readings_array[num_beams_per_col:]

        x1 = np.multiply(r_array, self.cache_theta_sin)
        x = np.multiply(x1, self.cache_phi_cos)

        y1 = np.multiply(r_array, self.cache_theta_sin)
        y = np.multiply(y1, self.cache_phi_sin)

        z = np.multiply(r_array, self.cache_theta_cos)

        readings = np.stack((x, y, z, intensity_array), axis=1)
        return readings
