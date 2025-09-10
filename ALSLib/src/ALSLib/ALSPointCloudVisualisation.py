import threading, time
import numpy as np
import open3d as o3d
from . import ALSHelperLidarLibrary as ALSLidar


def get_o3d_points(point_array):
    o3d_points = point_array[0:, 0:3]
    o3d_points = o3d.utility.Vector3dVector(np.asarray(o3d_points))
    return o3d_points


def get_o3d_points_and_intensities(point_array):
    o3d_points = point_array[0:, 0:3]
    intensities = [x[3] for x in point_array]
    o3d_points = o3d.utility.Vector3dVector(np.asarray(o3d_points))
    return o3d_points, intensities


class ALSPointCloudVisualiser(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.pcd = o3d.geometry.PointCloud()
        self.visualiser = o3d.visualization.VisualizerWithKeyCallback()
        self.should_add_geometry = False
        self.should_update_geometry = False
        self.should_update_camera = True
        self.angle = 0
        self.stop_vis = False
        self.vis_init = True
        self.start()
        self.mesh = o3d.geometry.TriangleMesh
        self.linemesh = o3d.geometry.LineSet
        self.hasMesh = False
        self.should_remove_geometry = False
        self.movedMesh = o3d.geometry.LineSet
        self.meshRotation = (0, 0, 0)
        self.meshTranslation = (0, 0, 0)
        self.meshScale = (0, 0, 0)
        self.transformMesh = False
        self.hasBoxes = False
        self.boxes = []
        self.boxesToRemove = []
        self.initialisedBoxes = 0
        self.spawnBoxes = False
        self.added_mesh = False

    def add_geometry(self, points):
        print("Add geometry")
        self.pcd.points = points
        self.should_add_geometry = True

    def update_geometry(self, points):
        if self.vis_init:
            self.pcd.points = points
            self.should_add_geometry = True
            self.vis_init = False
        else:
            # print("Update geometry")
            self.pcd.points = points
            self.should_update_geometry = True

    def update_geometry_w_colors(self, points, colors):
        colors = o3d.utility.Vector3dVector(np.asarray(colors))

        if self.vis_init:
            self.pcd.points = points
            self.pcd.colors = colors
            self.should_add_geometry = True
            self.vis_init = False
        else:
            # print("Update geometry")
            self.pcd.points = points
            self.pcd.colors = colors
            self.should_update_geometry = True

    def add_mesh(self, mesh):
        self.mesh = mesh
        self.added_mesh = True
        self.should_add_geometry = True

    def transform(self, rotation, translation, scale):
        self.meshRotation = rotation
        self.meshTranslation = translation
        self.meshScale = scale
        # self.linemesh = self.linemesh.rotate(self.linemesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4)))
        self.transformMesh = True
        # self.should_update_geometry = True

    def add_box(self, center, rotation, size):
        self.hasBoxes = True
        bbox = o3d.geometry.OrientedBoundingBox(center, rotation, size)
        bbox.color = [1, 0, 0]
        self.boxes.append(bbox)

    def add_box_from_points(self, points):
        self.hasBoxes = True
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(points)
        )
        bbox.color = [0, 1, 0]
        self.boxes.append(bbox)

    def spawn_in_boxes(self):
        self.spawnBoxes = True

    def stop(self):
        self.stop_vis = True

    def remove_mesh(self):
        self.should_update_geometry = True
        self.should_remove_geometry = True

    def run(self):
        print("Run visualiser")
        self.visualiser.create_window("Point Cloud Visualisation")
        view_control = self.visualiser.get_view_control()

        while not self.stop_vis:
            if self.should_add_geometry:
                self.visualiser.add_geometry(self.pcd)
                if not self.hasMesh and self.added_mesh:
                    self.linemesh = o3d.geometry.LineSet.create_from_triangle_mesh(
                        self.mesh
                    )
                    self.linemesh.paint_uniform_color([0.8, 0.8, 0.8])
                    self.visualiser.add_geometry(self.linemesh, False)
                    self.visualiser.update_geometry(self.linemesh)
                    self.hasMesh = True
                if self.hasBoxes and self.spawnBoxes:
                    for box in self.boxes:
                        self.visualiser.add_geometry(box, False)
                    self.spawnBoxes = False
                    self.boxesToRemove = self.boxes
                    self.boxes = []
                self.should_add_geometry = False
                view_control.set_front(np.array([1, 0, 0]))
                self.visualiser.update_renderer()

            elif self.should_update_geometry:
                self.visualiser.update_geometry(self.pcd)
                if self.should_remove_geometry:
                    self.visualiser.remove_geometry(self.linemesh, False)
                    self.visualiser.update_geometry(self.linemesh)
                # if self.hasMesh:
                # self.visualiser.update_geometry(self.linemesh)
                if self.hasBoxes and self.spawnBoxes:
                    for box in self.boxesToRemove:
                        self.visualiser.remove_geometry(box, False)
                    for box in self.boxes:
                        self.visualiser.add_geometry(box, False)
                    self.spawnBoxes = False
                    self.boxesToRemove = self.boxes
                    self.boxes = []
                if self.transformMesh:
                    self.linemesh.scale(self.meshScale, center=(0, 0, 0))
                    self.linemesh.rotate(self.meshRotation)
                    self.linemesh.translate(self.meshTranslation)
                    self.visualiser.update_geometry(self.linemesh)
                    self.visualiser.get_view_control().set_up((1, 0, 0))
                    self.visualiser.get_view_control().set_front((0, -1, 1))
                    self.visualiser.get_view_control().set_zoom(1.5)
                    self.transformMesh = False
                self.should_update_geometry = False
                self.visualiser.update_renderer()

            view_control.set_up(np.array([0, 0, 1]))
            self.visualiser.poll_events()
            time.sleep(0.0333)
            if not threading.main_thread().is_alive():
                break

        self.visualiser.destroy_window()


class FrameBuilder:
    def __init__(self):
        self.point_array = []
        self.conv = ALSLidar.convert_distance_to_3d()
        self.vis = ALSPointCloudVisualiser()
        self.point_array = []
        self.num_points_in_frame = 0
        self.has_cached_angles = False

    """
    def add_column_points(self, num_beams_per_col, ColId, readings_array, v_span_degree = 30, h_span_degree = 1024):

        h_steps = 360/h_span_degree #this is from config files o fScannerAngleDistance
        v_steps = v_span_degree/num_beams_per_col #this is from config files o fScannerAngleDistance
        #v_span_degree = 30

        phi = np.deg2rad(180-h_steps*ColId)
        self.conv.cache_phi(phi)

        if not self.has_cached_angles:
            for count, d in enumerate(readings_array):
                theta = np.deg2rad(90-(v_steps*(count-num_beams_per_col*0.5)))
                self.conv.cache_theta(theta)

                if count>=num_beams_per_col-1: # after the list of distances we get the list of intensities
                    self.has_cached_angles = True
                    break

        self.point_array.extend(self.conv.distance_to_3d(readings_array, num_beams_per_col))
    """

    def add_column_points(self, num_beams_per_col, ColId, readings_array, lidar_info):
        if "beam_altitude_angles" in lidar_info:
            assert num_beams_per_col == len(lidar_info["beam_altitude_angles"])
        h_steps = lidar_info["span_width_angle_degree"] / lidar_info["span_width_steps"]

        phi_r = np.deg2rad(180 - h_steps * ColId)
        self.conv.cache_phi(phi_r)

        if not self.has_cached_angles:
            for count, d in enumerate(readings_array):
                if "beam_altitude_angles" in lidar_info:
                    angle = lidar_info["beam_altitude_angles"][count]
                else:
                    angle = (
                        lidar_info["span_height_angle_degree"]
                        / num_beams_per_col
                        * (count - num_beams_per_col * 0.5)
                    )
                theta_r = np.deg2rad(90 - angle)
                self.conv.cache_theta(theta_r)

                if (
                    count >= num_beams_per_col - 1
                ):  # after the list of distances we get the list of intensities
                    self.has_cached_angles = True
                    break

        self.point_array.extend(
            self.conv.distance_to_3d(readings_array, num_beams_per_col)
        )

    def display_frame(self):
        self.vis.update_geometry(get_o3d_points(np.reshape(self.point_array, (-1, 4))))

        self.point_array = []
        self.num_points_in_frame = 0
        self.conv.clear_cached()
        self.has_cached_angles = False
