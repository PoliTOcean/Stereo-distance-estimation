import cv2
from enum import Enum

class States(Enum):
    INIT             = 0
    KEYPOINTS        = 1
    SELECTION        = 2
    MATCHING         = 3
    ESTIMATION       = 4
    COMPUTE_DISTANCE = 5

class Scene:
    def __init__(self, left_img, right_img):
        self.keypoints = []
        self.selected_points = []
        self.distance_points = []
        self.nnpoint = []

        self.actions = []
        self.lines = []

        self.left_img = left_img
        self.right_img = right_img
        self.img_vis = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
        self.copy_img = self.img_vis.copy()

        self.status = States.INIT

    def draw_point(self, center_coords, r, color):
        # image, center_coordinates, radius, color, thickness
        cv2.circle(self.img_vis, center_coords, r, color, -1)
        self.actions.append((center_coords, r, color))

    def delete_point(self):
        if len(self.actions):
            self.img_vis = self.copy_img.copy()
            self.actions.pop()
            past_actions = self.actions.copy()
            self.actions = []
            
            for point in past_actions:
                self.draw_point(point[0], point[1], point[2])

    def draw_line(self, pt1, pt2, color):
        cv2.line(self.img_vis, pt1, pt2, color, 1)
        self.lines.append((pt1, pt2, color))

    def delete_line(self):
        if len(self.lines):
            self.img_vis = self.copy_img.copy()
            self.lines.pop()
            past_lines = self.lines.pop()
            self.lines = []

        for line in past_lines:
            self.draw_line(line[0], line[1], line[2])
