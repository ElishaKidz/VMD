import numpy as np

from KLTWrapper import KLTWrapper
from MovingCameraVMD.utiles import get_grid_coordinates, get_prev_centers, overlap
from Grid import Grid


class ForegroundPicker:
    def __init__(self, grid_size=4, theta_d=35, theta_v=50 * 50, theta_s=2, init_age=1, init_var=20*20, truncate_age=30,
                 lam=0.001, return_probs=False):
        self.klt = KLTWrapper()
        self.coords = []
        self.grids = []

        self.grid_size = grid_size
        self.theta_d = theta_d
        self.theta_s = theta_s
        self.theta_v = theta_v
        self.init_age = init_age
        self.init_var = init_var
        self.truncate_age = truncate_age
        self.lam = lam
        self.return_probs = return_probs

        self.num_frames = 0
        self.frame_size = None
        self.grid_area = None

    def create_grids(self, gray_frame):
        h, w = gray_frame.shape
        self.frame_size = gray_frame.shape
        if h % self.grid_size or w % self.grid_size:
            raise IOError("image size should divide by grid size")
        self.grid_area = h * w // (self.grid_size ** 2)
        self.coords = get_grid_coordinates(self.grid_size, gray_frame.shape)
        for coord in self.coords:
            x0, y0, x1, y1 = coord
            g = Grid(x0, y0, x1, y1, gray_frame[y0:y1, x0:x1], self.lam, self.theta_d, self.theta_v, self.init_age,
                     self.theta_s)
            self.grids.append(g)

    def compensate(self, prev_centers):
        h_, w_ = self.frame_size[0] // (2*self.grid_size), self.frame_size[1] // (2*self.grid_size)
        for prev_center, now_grid in zip(prev_centers, self.grids):
            x, y, _ = prev_center
            prev_grid_coords = [x - w_ + 1, y - h_ + 1, x + w_, y + h_]
            overlap_weights = []
            overlap_means = []
            overlap_vars = []
            overlap_ages = []
            count_overlaps = 0
            for grid in self.grids:
                w = overlap(prev_grid_coords, grid.get_coords()) / self.grid_area   # TODO: fix tiny overlaps #TODO: find out if fixed
                if w > 0:
                    overlap_weights.append(w)
                    mn, vr, age = grid.get_prev_params()
                    overlap_means.append(mn)
                    overlap_vars.append(vr)
                    overlap_ages.append(age)
                    count_overlaps += 1
                    if count_overlaps >= 4 or sum(overlap_weights) >= 1:  # save time
                        break
            if len(overlap_weights) < 4:
                overlap_weights = [w/sum(overlap_weights) for w in overlap_weights]  # normalize so sum is 1
            now_grid.compensate_model(overlap_weights, overlap_means, overlap_vars, overlap_ages)

    def final_update_grid_models(self):
        for grid in self.grids:
            grid.choose_update_and_swap_models()

    def decide(self):
        decision = np.zeros(self.frame_size)
        for grid in self.grids:
            coords = grid.get_coords()
            if self.return_probs:
                decision[coords[1]:coords[3], coords[0]:coords[2]] = grid.get_prob_mat()
            else:
                decision[coords[1]:coords[3], coords[0]:coords[2]] = grid.get_threshold_mat()
        return decision.astype(np.uint8)

    def pick_foreground(self, gray_frame):
        self.num_frames += 1

        if self.num_frames == 1:
            self.create_grids(gray_frame)
            self.klt.init(gray_frame)
            return np.zeros_like(gray_frame)

        homography = self.klt.RunTrack(gray_frame)

        for coord, grid in zip(self.coords, self.grids):
            grid.update_values(gray_frame[coord[1]:coord[3], coord[0]:coord[2]])

        centers = [grid.center for grid in self.grids]
        prev_centers = get_prev_centers(homography, centers)
        self.compensate(prev_centers)
        self.final_update_grid_models()
        d = self.decide()
        return d


