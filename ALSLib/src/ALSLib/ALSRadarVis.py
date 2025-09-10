from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QOpenGLWidget, QSlider
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPainter
from OpenGL.GL import *
import sys, os
import numpy as np
import json
from .ALSHelperFunctionLibrary import get_sensordata_path, get_lib_resource


class MainWindow(QMainWindow):
    def __init__(
        self,
        save_images=True,
        winsizex=800,
        winsizey=600,
        size_margin=400,
        min_range=0.1,
        max_range=9.0,
        default_range=0.5,
        min_gain=1,
        max_gain=10,
        default_gain=1,
        min_pointsize=1,
        max_pointsize=20,
        min_lifetime=5,
        max_lifetime=100,
        default_lifetime=15,
    ):

        super().__init__()
        self.image_need_save = False
        self.save_images = save_images
        self.controls_margin_size = size_margin
        self.image_ID = 0
        self.image_sensor_pos = ""
        self.image_timestamp = ""
        self.ui_text_stylesheet = "color: #00EE00"
        self.slider_stylesheet = get_lib_resource("ALSRadarVisSliderStyle")

        self.setGeometry(300, 300, winsizex, winsizey)
        self.setWindowTitle("Sensor Visualization")
        self.setStyleSheet("background-color: black;")

        self.left_margin = size_margin
        self.sensor_sizex = round(winsizex)
        self.sensor_sizey = round(winsizey)

        self.radar = Sensor(
            self.sensor_sizex, self.sensor_sizey, default_range, parent=self
        )
        self.radar.move(self.controls_margin_size, 0)
        self.radar.resize(self.sensor_sizex, self.sensor_sizey)

        self.radar.gain_multiplier = default_gain
        self.vertical_offset = 0
        self.range_slider, self.range_slide_val = self.add_slider(
            "Range:",
            self.zoom_slider_value_changed,
            min_range,
            max_range,
            default_range,
            "nmi",
            5,
        )
        self.gain_slider, self.gain_slider_val = self.add_slider(
            "Gain:", self.gain_slider_value_changed, min_gain, max_gain, default_gain
        )
        self.size_slider, self.size_slider_val = self.add_slider(
            "Size:",
            self.pointsize_slider_value_changed,
            min_pointsize,
            max_pointsize,
            self.radar.size_multiplier,
            "px",
        )
        self.lifetime_slider, self.lifetime_slider_val = self.add_slider(
            "LifeTime:",
            self.lifetime_slider_value_changed,
            min_lifetime,
            max_lifetime,
            default_lifetime,
            "frames",
        )
        self.ringdist_slider, self.ringdist_slider_val = self.add_slider(
            "RingDist:", self.ringrange_slider_value_changed, 0.1, 10, 1, "nmi"
        )

        self.update_needed = True
        timer = QtCore.QTimer(self)
        timer.setInterval(15)
        timer.timeout.connect(self.update_if_required)
        timer.start()

    def update_if_required(self):
        if self.update_needed:
            self.radar.updateGL()
            self.update_needed = False

    def zoom_slider_value_changed(self, new_value):
        self.radar.scale = new_value / 100
        self.range_slide_val.setText(str(new_value / 100) + " nmi")
        self.radar.makeCurrent()
        self.radar.resizeGL(self.sensor_sizex, self.sensor_sizey)
        self.radar.doneCurrent()
        self.update_needed = True

    def gain_slider_value_changed(self, new_value):
        self.radar.gain_multiplier = new_value / 100
        self.gain_slider_val.setText(str(new_value / 100))
        self.update_needed = True

    def lifetime_slider_value_changed(self, new_value):
        self.radar.lifetime = new_value / 100
        self.lifetime_slider_val.setText(str(new_value / 100) + " frames")
        self.update_needed = True

    def pointsize_slider_value_changed(self, new_value):
        self.radar.size_multiplier = new_value / 100
        self.size_slider_val.setText(str(new_value / 100) + " px")
        self.update_needed = True

    def ringrange_slider_value_changed(self, new_value):
        self.radar.ring_dist = new_value / 100
        self.ringdist_slider_val.setText(str(new_value / 100) + " nmi")
        self.update_needed = True

    def add_slider(
        self, title, func, rangemin, rangemax, default_val, unit="", range_step=1
    ):
        self.label = QLabel(self)
        self.label.setText(title)
        self.label.setStyleSheet(self.ui_text_stylesheet)
        font = QFont()
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.move(0, self.vertical_offset)

        slider = QSlider(Qt.Horizontal, self)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.NoTicks)
        slider.setStyleSheet(self.slider_stylesheet)

        slider.setTickInterval(10)
        slider.setSingleStep(range_step)
        slider.setRange(round(rangemin * 100), round(rangemax * 100))
        slider.setValue(round(default_val * 100))
        slider.move(140, self.vertical_offset)
        slider.valueChanged[int].connect(func)

        label2 = QLabel(self)
        label2.setText(str(slider.value() / 100) + " " + unit)
        label2.setStyleSheet(self.ui_text_stylesheet)
        font = QFont()
        font.setPointSize(16)
        label2.setFont(font)
        label2.move(250, self.vertical_offset)
        label2.resize(120, 32)

        self.vertical_offset += 32

        return slider, label2

    def resizeEvent(self, event):
        self.resize_radar(self.size().width(), self.size().height())
        QMainWindow.resizeEvent(self, event)

    def resize_radar(self, w, h):
        self.sensor_sizex = w - self.left_margin
        self.sensor_sizey = h
        self.radar.move(self.left_margin, 0)
        self.radar.resize(self.sensor_sizex, self.sensor_sizey)

    def save_radar_image(self):
        if self.save_images and self.image_need_save:
            # grab whole window, offset by controls margin size, to just include the radar image
            screenshot = self.radar.grab()

            # grab doesn't get the background, so we need to add it to the image before saving
            maskColor = QColor.fromRgbF(0.0, 0.0, 0.0, 1.0)
            mask = screenshot.createMaskFromColor(maskColor, Qt.MaskOutColor)
            p = QPainter(screenshot)
            p.setPen(QColor.fromRgbF(0.0, 0.05, 0.8, 1.0))
            p.drawPixmap(screenshot.rect(), mask, mask.rect())
            p.end()

            self.image_ID += 1
            datapath = get_sensordata_path("/RadarImages/")
            if not os.path.exists(datapath):
                os.makedirs(datapath)

            filename = (
                "img"
                + str(self.image_ID)
                + "_"
                + str(self.image_timestamp)
                + "_"
                + str(self.image_sensor_pos)
                + ".bmp"
            )
            screenshot.save(os.path.join(datapath, filename), "bmp")
            self.image_need_save = False

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        event.accept()


class Dot:
    def __init__(self, x, y, w, h):
        self.x = x * 0.5399568  # km to nautical miles conversion
        self.y = y * 0.5399568
        self.w = w
        self.h = h
        self.red = 1
        self.green = 1
        self.blue = 1
        self.alpha = 1
        self.total_lifetime = 5
        self.timer = self.total_lifetime
        self.destroy = False

    def set_rgb(self, red: float, green: float, blue: float):
        self.red = red
        self.green = green
        self.blue = blue

    def set_rgba(self, red: float, green: float, blue: float, alpha: float):
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha

    def draw(self):
        self.timer -= 1

        if self.timer < 0 or self.alpha <= 0:
            self.destroy = True
            return

        # put center of square to the x y location
        posx = (float(self.x) - float(self.w)) / 2
        posy = (float(self.y) - float(self.w)) / 2

        # make 2 triangles to make a square
        glColor4f(
            self.red,
            self.green,
            self.blue,
            self.alpha * (self.timer / self.total_lifetime),
        )
        glVertex3f(posx, posy, 0)
        glVertex3f(posx + self.w, posy + self.h, 0)
        glVertex3f(posx + self.w, posy, 0)
        glVertex3f(posx, posy, 0)
        glVertex3f(posx, posy + self.h, 0)
        glVertex3f(posx + self.w, posy + self.h, 0)

    def setPosition(self, x, y):
        self.x = x
        self.y = y

    def setSize(self, w, h):
        self.w = w
        self.h = h

    def setLifetime(self, lifetime):
        self.total_lifetime = lifetime
        self.timer = self.total_lifetime


class Sensor(QOpenGLWidget):

    def __init__(self, sizex, sizey, default_range, parent=None):
        super().__init__(parent)

        self.sizex = sizex
        self.sizey = sizey
        self.move_left = True
        self.scale = default_range
        self.gain_multiplier = 1
        self.size_multiplier = 4.5
        self.lifetime = 5
        self.dots = []
        self.min_range = 2
        self.max_range = 9
        self.draw_range_circles = False
        self.ring_dist = 1

        self.drawn_once = False

    def initializeGL(self):
        glClearColor(0.0, 0.05, 0.8, 0.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClear(GL_COLOR_BUFFER_BIT)

    def resizeGL(self, w, h):
        if w > 0 and h > 0:
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            aspect = float(w) / float(h)

            scale_multiplier = self.scale * 1000 * 1.852  # nautical miles to km

            glOrtho(
                -scale_multiplier * aspect / 2,
                scale_multiplier * aspect / 2,
                -scale_multiplier / 2,
                scale_multiplier / 2,
                -1,
                1,
            )

    def drawLoop(self):
        glClear(GL_COLOR_BUFFER_BIT)

        glColor4f(0, 0.6, 0.4, 1.0)
        glEnable(GL_LINE_SMOOTH)
        glLineWidth(2)

        if self.draw_range_circles:
            # glClear(GL_COLOR_BUFFER_BIT)
            glColor4f(0, 0.6, 0.4, 1.0)

            glEnable(GL_LINE_SMOOTH)
            glLineWidth(2)

            for i in range(0, int(self.max_range + 1)):
                self.draw_circle(i * self.ring_dist * 1000 * 1.852)

        self.draw_line(0, 0, 0, self.sizey * 100)

        glBegin(GL_TRIANGLES)
        i = 0
        while i < len(self.dots):
            if self.dots[i].destroy:
                self.dots.remove(self.dots[i])
            else:
                self.dots[i].draw()
                i += 1
        glEnd()

        self.drawn_once = True

    def paintGL(self):
        self.drawLoop()

    def updateGL(self):
        self.update()
        self.window().save_radar_image()

    def draw_circle(self, radius):

        glBegin(GL_LINE_LOOP)
        segments = 64
        for i in range(0, segments):

            theta = 2.0 * 3.1415926 * float(i) / float(segments)

            x = 0.5 * radius * np.cos(theta)
            y = 0.5 * radius * np.sin(theta)

            glVertex2f(x, y)
        glEnd()

    def draw_line(self, startx, starty, endx, endy):
        glBegin(GL_LINES)
        glVertex2f(startx, starty)
        glVertex2f(endx, endy)
        glEnd()


def read_from_data(data, variable_name, default):
    value = default
    if variable_name in data:
        value = data.get(variable_name)
    print(variable_name, ":", default, " . ", value)
    return value


class RadarVisualization:
    def __init__(
        self,
        data_queue,
        config_file_name: str = "ALSRadarVisualisationConfig.json",
        winsizex=800,
        winsizey=400,
        min_range=2.0,
        max_range=9.0,
        default_range=2.0,
        min_gain=1,
        max_gain=10,
        default_gain=1,
        min_pointsize=1,
        max_pointsize=20,
        min_lifetime=5,
        max_lifetime=100,
        default_lifetime=15,
    ):

        # Reading params from the config json
        vis_data = json.loads(get_lib_resource(config_file_name))

        Fullscreen = read_from_data(vis_data, "Fullscreen", False)
        SaveImages = read_from_data(vis_data, "SaveImages", False)
        ScreenIdx = read_from_data(vis_data, "ScreenIdx", 0)
        WindowSizeX = read_from_data(vis_data, "WindowSizeX", winsizex)
        WindowSizeY = read_from_data(vis_data, "WindowSizeY", winsizey)
        WindowSizeAuto = read_from_data(vis_data, "WindowSizeAuto", False)
        MarginSize = read_from_data(vis_data, "MarginSize", 400)
        MinRange = read_from_data(vis_data, "MinRange", min_range)
        MaxRange = read_from_data(vis_data, "MaxRange", max_range)
        DefaultRange = read_from_data(vis_data, "DefaultRange", default_range)
        MinGain = read_from_data(vis_data, "MinGain", min_gain)
        MaxGain = read_from_data(vis_data, "MaxGain", max_gain)
        DefaultGain = read_from_data(vis_data, "DefaultGain", default_gain)
        MinPointSize = read_from_data(vis_data, "MinPointSize", min_pointsize)
        MaxPointSize = read_from_data(vis_data, "MaxPointSize", max_pointsize)
        MinLifetime = read_from_data(vis_data, "MinLifetime", min_lifetime)
        MaxLifetime = read_from_data(vis_data, "MaxLifetime", max_lifetime)
        DefaultLifeTime = read_from_data(vis_data, "DefaultLifeTime", default_lifetime)
        DrawRangeCircles = read_from_data(vis_data, "DrawRangeCircles", True)

        self.app = QApplication(sys.argv)
        self.save_screenshot = False
        if Fullscreen:
            Screen = self.app.screens()[ScreenIdx]
            ScreenSize = Screen.size()

            self.window = MainWindow(
                SaveImages,
                ScreenSize.width(),
                ScreenSize.height(),
                MarginSize,
                MinRange,
                MaxRange,
                DefaultRange,
                MinGain,
                MaxGain,
                DefaultGain,
                MinPointSize,
                MaxPointSize,
                MinLifetime,
                MaxLifetime,
                DefaultLifeTime,
            )
            Geometry = Screen.geometry()
            self.window.move(Geometry.x(), Geometry.y())
            self.window.showFullScreen()
        else:

            if WindowSizeAuto:
                ScreenSize = self.app.screens()[ScreenIdx].size()
                WindowSizeX = round(ScreenSize.width() * 0.75)
                WindowSizeY = round(ScreenSize.height() * 0.75)
            self.window = MainWindow(
                SaveImages,
                WindowSizeX,
                WindowSizeY,
                MarginSize,
                MinRange,
                MaxRange,
                DefaultRange,
                MinGain,
                MaxGain,
                DefaultGain,
                MinPointSize,
                MaxPointSize,
                MinLifetime,
                MaxLifetime,
                DefaultLifeTime,
            )

        # start data loop
        self.data_queue = data_queue
        data_timer = QtCore.QTimer(self.window)
        data_timer.setInterval(5)
        data_timer.timeout.connect(self.update_readings)
        data_timer.start()

        self.window.radar.lifetime = DefaultLifeTime

        self.draw_range_circles(DrawRangeCircles)

        self.window.radar.min_range = MinRange
        self.window.radar.max_range = MaxRange

    def run(self):
        self.window.show()
        self.result = self.app.exec_()

    def update_readings(self):
        try:
            dot = self.data_queue.get_nowait()
            if dot:
                self.add_dot(dot)
                self.window.update_needed = True
                self.save_bmp()
        except Exception:
            pass

    def save_bmp(self, sensor_position="", timestamp=""):
        if self.window.radar.drawn_once:
            self.window.image_sensor_pos = sensor_position
            self.window.image_timestamp = timestamp
            self.window.image_need_save = True

    def shutdown(self):
        self.window.close()

    def add_dot(self, dot: Dot):
        dot.setLifetime(self.window.radar.lifetime)
        dot.set_rgba(
            dot.red, dot.green, dot.blue, 0.1 * self.window.radar.gain_multiplier
        )
        dot.setSize(
            dot.w * self.window.radar.size_multiplier,
            dot.h * self.window.radar.size_multiplier,
        )
        radar_widget = self.window.radar
        radar_widget.dots.append(dot)
        self.dots = radar_widget.dots

    def set_scale(self, scale):
        self.window.radar.scale = scale

    def clear_dots(self):
        self.window.radar.dots.clear()

    def draw_circle(self, radius, r, g, b):
        self.window.radar.draw_circle(radius, r, g, b)

    def draw_range_circles(self, bdraw_range_circles: bool):
        self.window.radar.draw_range_circles = bdraw_range_circles
