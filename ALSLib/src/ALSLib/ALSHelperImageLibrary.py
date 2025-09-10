from subprocess import Popen, PIPE
import math, cv2
import numpy as np
import numpy, cv2, json
from PIL import Image
import xml.etree.ElementTree as ET

#####################################################
#
#  some functions and classes useful for image sensor
# processing
#
# opencv is required for this implementation
#
#####################################################


class BrownParams:
    k1, k2, k3, p1, p2 = 0.0, 0.0, 0.0, 0.0, 0.0
    width, height, h_fov = 0, 0, 0
    distortion_coefs, camera_matrix = None, None

    def __init__(self):
        self.height = 0

    # ############################################
    #
    # helper to get the distorted coordinates of the 2d points sent inside the stream
    # uses exampe:
    # brown = BrownParams().Load2(parsed_string['BrownModel'], width, height, fov)
    # distorted_points = brown.DoUndistortPixel(points)
    # for point in distorted_points:
    # 		# do something
    # ############################################

    def Load(self, k1, k2, p1, p2, k3=0.0, width=720, height=480, h_fov=90):
        self.k1, self.k2, self.k3, self.p1, self.p2 = k1, k2, k3, p1, p2
        self.width, self.height, self.h_fov = width, height, h_fov
        self.UpdateCoefs()
        return self

    def Load2(self, jsondata, width=720, height=480, h_fov=90):
        self.k1 = float(jsondata["k1"])
        self.k2 = float(jsondata["k2"])
        self.k4 = float(jsondata["k3"])
        self.p1 = float(jsondata["p1"])
        self.p2 = float(jsondata["p2"])
        self.width, self.height, self.h_fov = width, height, h_fov
        self.UpdateCoefs()
        return self

    def UpdateCoefs(self):
        p = self.height / self.width
        f = (float(self.width) * 0.5) / math.tan(
            math.radians(float(self.h_fov) * 0.5)
        )  # Calculating the focal length from the FoV
        self.camera_matrix = np.array(
            [
                [f, 0.0, self.width * 0.5],
                [0.0, f * p, self.height * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.distortion_coefs = np.array(
            [self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32
        )

    def DoDistortPixel(self, coord):
        normCoord = (
            (float(coord[0]) / float(self.width)),
            (float(coord[1]) / float(self.height)),
        )
        normCoord = (normCoord[0] * 2.0 - 1.0, normCoord[1] * 2.0 - 1.0)
        r = math.dist(normCoord, (0.0, 0.0))
        x = float(normCoord[0])
        y = float(normCoord[1])

        r2, r4, r6 = pow(r, 2), pow(r, 4), pow(r, 6)

        # //f=1+k1*r^2 + k2*r^4 + k3*r^6
        f = 1.0 + (self.k1 * r2) + (self.k2 * r4) + (self.k3 * r6)
        ud = normCoord[0] * f
        vd = normCoord[1] * f

        # //fx = 2p1 * x * y+p2*(r^2+2*x^2)
        ud += (2.0 * self.p1 * x * y) + (self.p2 * (r2 + 2.0 * (x * x)))
        # //fy=p1*(r^2+2*y^2)+2*p2*x*y
        vd = self.p1 * (r2 + (2.0 * (y * y))) + (2.0 * self.p2 * x * y)

        ud = (ud + 1) * 0.5
        vd = (vd + 1) * 0.5
        return (int(ud * self.width), int(vd * self.height))

    def DoUndistortPixel(self, coord):
        inputpoints = np.float32(coord)
        dst = cv2.undistortPoints(
            inputpoints, self.camera_matrix, self.distortion_coefs, dst=None
        )
        output = dst.reshape(-1, 2)
        # ##method one
        # dst2 = dst[0][0]
        # dst2 = (dst2+1.0)*0.5
        # dst2 = round(dst2[0]*self.width), round(dst2[1]*self.height))

        ##method two (equivalent)
        # formula is : X=p[0]*fx+cx, Y =p[1]*fy+cy
        mat = np.delete(self.camera_matrix, 2, 0)
        points = np.insert(output, 2, 1, 1)
        out = np.stack(mat.dot(np.transpose(p)) for p in points)
        return out
        # fx = self.camera_matrix[0, 0]
        # fy = self.camera_matrix [1, 1]
        # cx = self.camera_matrix[ 0, 2]
        # cy = self.camera_matrix[1, 2]
        # for i, p in enumerate(output):
        # 	output[i] = (p[0]*fx+cx, p[1]*fy+cy)
        # return output

    def DoDistortImage(self, src_img):
        dst = cv2.undistort(src_img, self.camera_matrix, self.distortion_coefs)
        return dst


#
# Displays an image in another window until next one comes
#
def JustDisplay(image, showImage=True):
    if showImage:
        try:
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("img", image)
            cv2.waitKey(1)
        except Exception as e:
            print("Unable to show image: " + str(e))


def DecodeImageData(image_data, image_width, image_height, image_channels):
    array = np.frombuffer(image_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image_height, image_width, image_channels))
    return array


#
# conveniently add multiple images to the same file
#
def StackImages(array_of_images, num_per_line=2):
    num_images = len(array_of_images)
    if num_images < 1:
        return None
    if num_images == 1:
        return array_of_images[0]

    total_h, total_w = 0, 0
    current_h, current_w = 0, 0
    for i in range(len(array_of_images)):
        if i % num_per_line == 0:
            if current_w > total_w:
                total_w = current_w
            total_h += current_h
            current_h, current_w = 0, 0

        h, w = array_of_images[i].shape[0], array_of_images[i].shape[1]
        current_w += w
        if h > current_h:
            current_h = h

    if current_w > total_w:
        total_w += current_w

    total_h += current_h
    final_image = np.zeros((total_h, total_w, 4), dtype=np.uint8)
    current_h, current_w, total_h = 0, 0, 0
    for i in range(len(array_of_images)):
        if i % num_per_line == 0:
            total_h += current_h
            current_h, current_w = 0, 0

        image = array_of_images[i]
        h, w = image.shape[0], image.shape[1]
        end = total_h + h
        start = total_h
        final_image[start:end, current_w : current_w + w, :] = image
        current_w += w
        if h > current_h:
            current_h = h

    return final_image


class ImageWriter_Interface:
    def __init__(self, filename: str):
        self.filename = filename

    def add_image(self, image):
        return

    def close(self):
        return


class ImageWriterCompressedLoose(ImageWriter_Interface):
    def __init__(self, filename: str):
        super().__init__(filename)
        self.imageID = 0

    def add_image(self, image):
        out_file_name = self.filename + "_" + self.imageID + ".png"
        self.imageID += 1
        cv2.imwrite(out_file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])


class ImageWriterVideoStream(ImageWriter_Interface):
    def __init__(self, filename: str, FPS: int = 30):
        super().__init__(filename)
        self.FPS = FPS
        self.video_streaming_process = Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "image2pipe",
                "-r",
                str(FPS),
                "-i",
                "-",
                "-vcodec",
                "mpeg4",
                "-q:v",
                "5",
                "-b:v",
                "100M",
                "-r",
                str(FPS),
                filename,
            ],
            stdin=PIPE,
        )

    def add_image(self, image):
        image.save(self.video_streaming_process.stdin, "png")

    def add_image_from_array(self, image_array):
        from PIL import Image

        im = Image.fromarray(image_array)
        b, g, r, a = im.split()
        im = Image.merge("RGB", (r, g, b))
        im.save(self.video_streaming_process.stdin, "png")

    def close(self):
        self.video_streaming_process.stdin.close()
        self.video_streaming_process.wait()


def Parse2DBox(data_dict):
    return (
        round(data_dict[0]["X"]),
        round(data_dict[0]["Y"]),
        round(data_dict[1]["X"]),
        round(data_dict[1]["Y"]),
    )


def ParsePose(data_dict):
    pose_data = []
    for p in data_dict:
        pose_data.append((round(p["X"]), round(p["Y"]), p["Occ"]))
    return pose_data


def draw_one_box(reading, image, color, thickness, text: str = ""):
    x, y, w, h = Parse2DBox(reading)
    colorrec = (int(color[2]), int(color[1]), int(color[0]))  # BGR
    image = cv2.rectangle(image, (x, y), (w, h), colorrec, thickness)

    if text != "":
        font = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.putText(image, text, (x, y - 3), font, 0.3, colorrec, 1)


def draw_one_skeleton_joints(reading, image, color, thickness):
    skeleton = ParsePose(reading)
    for x, y, occluded in skeleton:
        if occluded:
            color = (0, 0, 255)  # BGR
        else:
            color = (0, 255, 0)  # BGR
        image = cv2.circle(image, (x, y), 2, color)


def draw_one_skeleton_sticknan(reading, image, color, thickness):
    skeleton = ParsePose(reading)

    color = [255, 0, 0]  # BGR
    for i in range(5):
        (xa, ya, oa), (xb, yb, ob) = skeleton[i], skeleton[1 + i]
        image = cv2.line(image, (xa, ya), (xb, yb), color, 1)
        color[2] += 60
    color = [0, 255, 0]  # BGR
    for i in range(6, 11):
        (xa, ya, oa), (xb, yb, ob) = skeleton[i], skeleton[1 + i]
        image = cv2.line(image, (xa, ya), (xb, yb), color, 1)
        color[2] += 60

    (xa, ya, oa), (xb, yb, ob) = skeleton[2], skeleton[8]
    image = cv2.line(image, (xa, ya), (xb, yb), color, 1)
    (xa, ya, oa), (xb, yb, ob) = skeleton[9], skeleton[3]
    image = cv2.line(image, (xa, ya), (xb, yb), color, 1)

    if len(skeleton) == 18:  # add eyes and hears
        (xa, ya, oa), (xb, yb, ob) = skeleton[12], skeleton[15]
        image = cv2.line(image, (xa, ya), (xb, yb), color, 1)
        for i in range(13, 17):
            color = [0, 0, 255]  # BGR
            (xa, ya, oa), (xb, yb, ob) = skeleton[i], skeleton[1 + i]
            image = cv2.line(image, (xa, ya), (xb, yb), color, 1)
            color[2] += 60
    else:
        (xa, ya, oa), (xb, yb, ob) = skeleton[12], skeleton[13]
        image = cv2.line(image, (xa, ya), (xb, yb), color, 1)


def GetColorArrayPNG(palette_image: str):
    im = cv2.imread(palette_image)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im[0]


# ----------------These functions are used for drawing bounding boxes on segmented images-------------
def GetColorMap(palette_image: str):
    im = Image.open(palette_image, "r")
    width, height = im.size
    pixel_values = list(im.getdata())
    pixel_values = numpy.array(pixel_values).reshape((width, height, 3))
    return pixel_values


def GetColorID(color, pixel_values):
    i = 0
    bestI = 0
    bestDist = 999

    col = np.where((pixel_values[:, 0, :] == color).all(axis=1))[0]
    if len(col) > 0:
        return col[0]
    else:
        print(f"No matching colors to {color} found")

    for col in pixel_values:
        if (col[0] == color).all():
            return i
        dist = numpy.absolute((col[0] - color)).sum()
        if dist <= 1:
            print("Near miss color: ", color)
            return i
        if dist < bestDist:
            bestI = i
            bestDist = dist
        i += 1
    print(
        "Color Not Found: ",
        color,
        " using ",
        pixel_values[bestI],
        " (dist:",
        bestDist,
        ")",
    )
    return bestI


def MapColorPalette(xml_file: str):
    mytree = ET.parse(xml_file)
    myroot = mytree.getroot()

    categories_to_colors = {}
    ordered_categories = []  # array is enough, ids are the key to the category name
    for i in range(256):
        ordered_categories.append("Unknown")

    for rule in myroot.iter("Rule"):
        colorID = rule.attrib.get("colorID")
        colorRange = rule.attrib.get("colorRange")
        color_start = color_end = 0
        if colorID:
            color_start = color_end = int(colorID)
        if colorRange:
            color_start = int(colorRange.split(":")[0])
            color_end = int(colorRange.split(":")[1])
        while color_start <= color_end:
            category = rule.attrib.get("category")
            if not category:
                print("error, no categry found in the rule: ", rule)
                continue
            ordered_categories[color_start] = category
            if categories_to_colors.get(category):
                if color_start not in categories_to_colors[category]:
                    categories_to_colors[category].append(color_start)
            else:
                categories_to_colors[category] = [color_start]
            color_start += 1

    return categories_to_colors, ordered_categories


def GetBoxesFromImage(mask):
    box_list = []
    shaped = mask.reshape(-1, mask.shape[2])
    color_list = numpy.unique(shaped, axis=0)
    for color in color_list:
        lower = numpy.array(color, dtype="uint8")
        upper = numpy.array(color, dtype="uint8")
        img_mask = cv2.inRange(mask, lower, upper)
        x, y, w, h = cv2.boundingRect(img_mask)

        count = numpy.sum(numpy.all(mask == color, axis=2))
        box_list.append((x, y, w, h, color, int(count)))
    return box_list
