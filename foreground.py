import numpy as np
import cv2 as cv
from skimage.util import view_as_windows
from VMD.MovingCameraForegroundEstimetor.ForegroundEstimetor import ForegroundEstimetor
from time import time
from numba import jit, prange
from collections import deque
import torch
foreground_estimators = {}


def register(name):
    def register_func_fn(cls):
        foreground_estimators[name] = cls
        return cls

    return register_func_fn


@register("MovingCameraForegroundEstimetor")
class MovingCameraForegroundEstimetor(ForegroundEstimetor):
    def __init__(self, num_models=2, block_size=4, var_init=20.0 * 20.0, var_trim=5.0 * 5.0, lam=0.001,
                 theta_v=50.0 * 50.0, age_trim=30, theta_s=2, theta_d=2, dynamic=False, calc_probs=False,
                 sensitivity="mixed", suppress=False, smooth=True):
        super(MovingCameraForegroundEstimetor, self).__init__(num_models, block_size, var_init, var_trim, lam, theta_v,
                                                              age_trim, theta_s, theta_d, dynamic, calc_probs,
                                                              sensitivity, suppress, smooth)


@register("MedianForegroundEstimation")
class MedianForegroundEstimation:
    def __init__(self, num_frames=10) -> None:
        self.frames_history = []
        self.num_frames = num_frames

    def __call__(self, frame):
        if len(self.frames_history) == 0:
            foreground = frame

        else:
            background = np.median(self.frames_history, axis=0).astype(dtype=np.uint8)
            foreground = cv.absdiff(frame, background)

            if len(self.frames_history) >= self.num_frames:
                self.frames_history = list(self.frames_history[-self.num_frames:])

        self.frames_history.append(frame)
        return foreground

    def reset(self):
        self.frames_history = []




@register("MOG2")
class MOG2():
    def __init__(self, **kwargs):
        self.fgbg = cv.createBackgroundSubtractorMOG2(**kwargs)
    
    def __call__(self, frame):
        fgmask = self.fgbg.apply(frame)
        return fgmask


@register("PESMODForegroundEstimation")
class PESMODForegroundEstimation():
    def __init__(self, neighborhood_matrix: tuple = (3, 3), num_frames=10, suppress=False) -> None:
        self.neighborhood_matrix = neighborhood_matrix
        self.frames_history = deque()
        self.num_frames = num_frames
        self.suppress = suppress
        self.filter_w, self.filter_h = self.neighborhood_matrix
        self.pad_w = int(self.filter_w / 2)
        self.pad_h = int(self.filter_h / 2)

        PESMODForegroundEstimation.compile_difference()

    @staticmethod
    def compile_difference():
        temp1 = np.zeros((400, 400, 3, 3), dtype=np.float64)
        temp2 = np.zeros((3, 3, 400, 400), dtype=np.uint8)
        difference(temp1, temp2)

    def __call__(self, frame):
        if len(self.frames_history) == 0:
            self.frames_history.append(frame)
            self.window_sum = self.frames_history[-1].astype(np.int32)
            foreground = frame
        else:
            background = self.window_sum / len(self.frames_history)
            padded_background = np.pad(background, ((self.pad_w, self.pad_w), (self.pad_h, self.pad_h)),
                                       constant_values=(np.inf, np.inf))
            background_patches = view_as_windows(padded_background, self.neighborhood_matrix)

            w, h = frame.shape

            broadcasted_frame = np.broadcast_to(frame, (self.filter_w, self.filter_h, w, h))

            foreground = difference(background_patches, broadcasted_frame)

            if self.suppress:
                mn = np.mean(frame)
                std = np.std(frame)
                foreground[frame < mn + std] = 0

            self.frames_history.append(frame)

            if len(self.frames_history) > self.num_frames:
                self.window_sum -= self.frames_history[0]
                self.frames_history.popleft()

            self.window_sum += self.frames_history[-1]
        return foreground

    def reset(self):
        self.frames_history = deque()


@jit(nopython=True, parallel=True)
def difference(background_patches, broadcasted_frame):
    h, w = broadcasted_frame.shape[2], broadcasted_frame.shape[3]

    frame = np.transpose(broadcasted_frame, (2, 3, 0, 1))

    foreground = np.empty((h, w), dtype=np.uint8)

    for i in prange(h):
        for j in prange(w):
            diff = np.abs(np.subtract(background_patches[i, j], frame[i, j]))
            min_diff = np.min(diff)
            foreground[i, j] = min_diff
    return foreground


@register("MGDForegroundEstimation")
class MGDForegroundEstimation():
    def __init__(self, kernel_shape=9):
        assert kernel_shape % 2 != 0, ValueError("The kernel should be odd")
        self.kernel_shape = kernel_shape
        self._kernel_shape_with_bg = kernel_shape + 2
        self.n_rings = kernel_shape // 2
        self.r = np.arange(1, self.n_rings)

        distance_from_center = list(range(1,(self._kernel_shape_with_bg//2)+1))
        self._X_distances = np.array([distance_from_center[::-1] + [0] + distance_from_center] * self._kernel_shape_with_bg)
        self._Y_distances = self._X_distances.transpose()
        self._distance_kernel = np.round(np.sqrt((self._X_distances)**2 + (self._Y_distances)**2))
        self.ring_kernels = np.stack([self.build_ring_kernel(self._distance_kernel,i) for i in range(self.n_rings)],axis=0)
        self.backgroung_kernel = self.build_ring_kernel(self._distance_kernel, self.n_rings+1)

        self._X_distances = self._X_distances[1:-1, 1:-1]
        self._Y_distances = self._Y_distances[1:-1, 1:-1]
    
    @staticmethod
    def build_ring_kernel(distance_kernel:np.array,requested_distance_from_center:int) -> np.array:
        """
        Create the ring convolution kernels as descrined in the article.
        set of connected pixels of the same distance.
        """  
        ring = np.zeros(distance_kernel.shape)
        ring[np.where(distance_kernel == requested_distance_from_center) ] = 1
        ring /= np.sum(ring)
        return ring
    
    @staticmethod
    def initial_difference_matrix(mu_matrices:np.array):
        shifted_mu_matrices = np.roll(mu_matrices,-1,axis=0) # [ring1,ring2,ring3,...,] -> [ring2,ring3,...,ring1]    
        mu_matrices[-1,:,:] = 0 # zero the last ring because its irrelevant to follwoing the substraction
        shifted_mu_matrices[-1,:,:] = 0 # zero the first ring because its irrelevant to the follwoing substraciton
        D = np.sum((np.maximum(mu_matrices - shifted_mu_matrices,0))**2,axis=0)
        return D
    
    @staticmethod
    def convolve_with_multiple_filters(gray_frame:np.array,filters:np.array,**kwargs):
        """
        gray_frame is expected to be an array of shape (height,width)

        filters are expected to be of shape (n_filters,kernel_height,kernel_width)

        """
        gray_frame = torch.tensor(gray_frame,dtype=torch.float32).view(1,1,*gray_frame.shape) # convert into (1,1,height,width)->(minibatch,channels,height,width)    
        filters = torch.tensor(filters, dtype=torch.float32).unsqueeze(1) # convert into (n_filters,1,kernel_height,kernel_width)->(out_channels,in_channels,kernel_height,kernel_width)
        output = torch.nn.functional.conv2d(gray_frame, filters,**kwargs).squeeze(0)
        return output.numpy() # return a numpy array as output
    
    def calculate_sigma(self, mu_matrices, B):
        P_r = (np.subtract(mu_matrices[1:], B))/(mu_matrices[0] - B) # p(r)-B/p(0)/B, 1<=r<=n_rings
        sigma_r = self.r/(np.sqrt(-2*np.log(P_r))) # refer algorithm 2 in the article to understand the line
        sigma_r[np.isnan(sigma_r)] = np.inf
        sigma_r[P_r == 0] = np.inf # if P_r equals zero it means that the ring is similar to the background and hence the corresponding sigma should be considered.
        return np.min(sigma_r,axis=0)
    
    def calculate_hessian_filter(self, sigma):
        exponent_power = np.exp(-(self._X_distances**2 + self._Y_distances**2)/(2*(sigma**2)))
        gxx_coefficient = (-1/(2*np.pi*(sigma**4))) * (1-((self._X_distances**2)/(sigma**2)))
        gyy_coefficient = (-1/(2*np.pi*(sigma**4))) * (1-((self._Y_distances**2)/(sigma**2)))
        gxy_coefficient = self._X_distances*self._Y_distances/(2*np.pi*(sigma**6))

        gxx = gxx_coefficient*exponent_power
        gyy = gyy_coefficient*exponent_power
        gxy = gxy_coefficient*exponent_power
        return np.array([gxx,gxy,gxy,gyy])
    
    @staticmethod
    def calculate_ratio_between_eiganvalues(ev1,ev2):
        ev1, ev2 = np.abs(ev1), np.abs(ev2)
        return min(ev1, ev2)/max(ev1, ev2)
    
    @staticmethod
    def calculate_hessian_eigenvalues(hessian_matrix):
        fxx, fxy, fyx, fyy = hessian_matrix.flatten()
        assert fxy == fyx, ValueError(f"Hessian matrix should be symmetric: {hessian_matrix}")
        diag_sum = fxx + fyy
        sqr_root = np.sqrt((fxx - fyy)**2 + 4*fxy*fyx)
        ev1 = (diag_sum + sqr_root)/2
        ev2 = (diag_sum - sqr_root)/2
        return ev1, ev2

    def __call__(self,gray_frame):
        mu_matrices = self.convolve_with_multiple_filters(gray_frame,self.ring_kernels,padding='same')
        B = self.convolve_with_multiple_filters(gray_frame,np.expand_dims(self.backgroung_kernel,axis=0),padding='same') # convert background_kernel into (1,kernel_height,kernel_width)
        D = self.initial_difference_matrix(mu_matrices)

        first_threshold, _ = cv.threshold(D.astype(np.uint16), 0.0, np.max(D), cv.THRESH_OTSU+cv.THRESH_BINARY)
        second_threshold, _ = cv.threshold(D[D > first_threshold].astype(np.uint16), 0.0, np.max(D), cv.THRESH_OTSU+cv.THRESH_BINARY)
        
        # zero all indices that doesnt surpass the threshold
        corrected_D = D.copy()
        corrected_D[corrected_D<=second_threshold] = 0

        number_of_points_from_filter_center = self.kernel_shape//2
        # zero the boundries
        boundries_map = np.ones(corrected_D.shape)
        boundries_map[number_of_points_from_filter_center:-number_of_points_from_filter_center,
                      number_of_points_from_filter_center:-number_of_points_from_filter_center] = 0
        
        corrected_D[boundries_map==1] = 0
        
        # Check the istropy for the points that surpassed the threshold
        for col, row in np.argwhere(corrected_D>second_threshold):
            sigma = self.calculate_sigma(mu_matrices[:, col, row], B[0, col, row])
            if sigma < np.inf:
                neighborhood = gray_frame[col-number_of_points_from_filter_center:col+number_of_points_from_filter_center+1,row-number_of_points_from_filter_center:row+number_of_points_from_filter_center+1]            

                hessian_filter = self.calculate_hessian_filter(sigma)                
                hessian_matrix = np.sum(neighborhood*hessian_filter,axis=(1,2)).reshape(2,2)
                ev1,ev2 = self.calculate_hessian_eigenvalues(hessian_matrix)
                
                if ev1<0 and ev2<0:
                    I = self.calculate_ratio_between_eiganvalues(ev1,ev2)
                    corrected_D[col,row] *= I
                else:
                    corrected_D[col,row] = 0
        
        corrected_D[corrected_D>second_threshold] = 255
        return corrected_D.astype(np.uint8)
