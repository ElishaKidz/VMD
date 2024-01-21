from numba import prange, jit
import numpy as np


@jit(nopython=True, parallel=True)
def reshape_to_2d_array_numba(arr: np.ndarray, new_shape: tuple):
    reshaped_array = np.empty(new_shape, dtype=arr.dtype)

    # Reshape the array without using np.reshape
    idx = 0
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            reshaped_array[i, j] = arr.flat[idx]
            idx += 1

    return reshaped_array


@jit(nopython=True)
def get_weights_for_directions(abs_offset_x: np.ndarray, abs_offset_y: np.ndarray, model_height: int, model_width: int):
    """
    get weight of horizontal offset, vertical offset, and diagonal offset
    :return: horizontal, vertical, and diagonal weights
    """
    W_H = reshape_to_2d_array_numba(abs_offset_x * (1 - abs_offset_y), (model_height, model_width))
    W_V = reshape_to_2d_array_numba(abs_offset_y * (1 - abs_offset_x), (model_height, model_width))
    W_HV = reshape_to_2d_array_numba(abs_offset_x * abs_offset_y, (model_height, model_width))
    W_self = reshape_to_2d_array_numba((1 - abs_offset_x) * (1 - abs_offset_y), (model_height, model_width))
    return W_H, W_V, W_HV, W_self


@jit(nopython=True)
def project(points: np.ndarray, H: np.ndarray):
    """
    project 3d points using projection matrix
    :param points: 3d points
    :param H: projection matrix
    :return: projected points
    """
    project_points = H.dot(points)
    new_w = project_points[2, :]
    new_x = (project_points[0, :] / new_w)  # current centers location in the previous frame center X
    new_y = (project_points[1, :] / new_w)  # current centers location in the previous frame center Y
    return new_x, new_y


@jit(nopython=True, parallel=True)
def project_parallel(points: np.ndarray, H: np.ndarray):
    n_points = points.shape[1]
    projected_points = np.empty((3, n_points), dtype=points.dtype)

    for i in prange(n_points):
        x, y, z = points[:, i]
        new_x = H[0, 0] * x + H[0, 1] * y + H[0, 2] * z + H[0, 3]
        new_y = H[1, 0] * x + H[1, 1] * y + H[1, 2] * z + H[1, 3]
        new_w = H[2, 0] * x + H[2, 1] * y + H[2, 2] * z + H[2, 3]

        projected_points[0, i] = new_x / new_w
        projected_points[1, i] = new_y / new_w
        projected_points[2, i] = new_w

    return projected_points[0], projected_points[1]


@jit(nopython=True)
def calculate_all_coords(prev_center_x, prev_center_y, block_size):
    prev_x_grid_coords_temp = prev_center_x / block_size
    prev_y_grid_coords_temp = prev_center_y / block_size

    prev_x_grid_coords = np.floor(prev_x_grid_coords_temp).astype(np.int32)  # grid location index X
    prev_y_grid_coords = np.floor(prev_y_grid_coords_temp).astype(np.int32)  # grid location index Y

    offset_x = prev_x_grid_coords_temp - prev_x_grid_coords - np.float32(0.5)  # offset of grid's center from the prev to curr frame X
    offset_y = prev_y_grid_coords_temp - prev_y_grid_coords - np.float32(0.5)  # offset of grid's center from the prev to curr frame Y

    abs_offset_x = np.abs(offset_x)  # offset of grid's center from the prev to curr frame X
    abs_offset_y = np.abs(offset_y)  # offset of grid's center from the prev to curr frame Y

    return prev_x_grid_coords, prev_y_grid_coords, offset_x, offset_y, abs_offset_x, abs_offset_y


@jit(nopython=True, parallel=True)
def update_by_condition(cond, temp, prev, W, x_grid_coords, y_grid_coords, grid_overlap_x, grid_overlap_y):
    num_temp_channels = temp.shape[0]
    num_grid_coords = y_grid_coords.shape[0]

    for i in prange(num_grid_coords):
        if cond[i]:
            for k in prange(num_temp_channels):
                y = grid_overlap_y[i]
                x = grid_overlap_x[i]
                temp[k, y_grid_coords[i], x_grid_coords[i]] += W[y_grid_coords[i], x_grid_coords[i]] * prev[k, y, x]
    return temp


@jit(nopython=True, parallel=True)
def update_var_by_condition(cond, temp, prev_vars, prev_means, means, W, x_grid_coords, y_grid_coords, prev_x,
                            prev_y):
    num_temp_channels = temp.shape[0]
    num_grid_coords = y_grid_coords.shape[0]
    for i in prange(num_grid_coords):
        if cond[i]:
            for k in prange(num_temp_channels):
                temp[k, y_grid_coords[i], x_grid_coords[i]] += W[y_grid_coords[i], x_grid_coords[i]] * \
                                                               (prev_vars[k, prev_y[i], prev_x[i]] +
                                                                np.power(means[k, y_grid_coords[i], x_grid_coords[i]] -
                                                                         prev_means[k, prev_y[i], prev_x[i]], 2))
    return temp


@jit(nopython=True, parallel=True)
def update_W_by_condition(W_sum, W, x_grid_cond, y_grid_cond, cond):
    num_temp_channels = W_sum.shape[0]
    num_grid_coords = x_grid_cond.shape[0]
    for i in prange(num_grid_coords):
        if cond[i]:
            for k in prange(num_temp_channels):
                W_sum[k, y_grid_cond[i], x_grid_cond[i]] += W[y_grid_cond[i], x_grid_cond[i]]
    return W_sum


@jit(nopython=True)
def compensate_mean_and_age(temp_means, temp_ages, prev_means, prev_ages, W, W_H, W_V, W_HV, W_self,
                            x_grid_coords, y_grid_coords, prev_grid_coords_x, prev_grid_coords_y, offset_x,
                            offset_y, model_height, model_width):
    """
    compensate the means and the ages of the model
    :param temp_means: where to temporarly store the compenated means
    :param temp_ages: same for ages
    :param prev_means: means of models in previous iterations
    :param prev_ages: same for ages
    :param W: sum of all weights
    :param W_H: horizontal weight
    :param W_V: vertical weight
    :param W_HV: diagonal weight
    :param W_self: weight of closest grid
    :param x_grid_coords: x grid coords from utils.get_grid_coords
    :param y_grid_coords: same for y
    :param prev_grid_coords_x: previouse coordinates of the grid
    :param prev_grid_coords_y: same for y
    :param offset_x: offset of grid's center from the prev to curr frame X
    :param offset_y: same for y
    :param model_height: model height
    :param model_width: model width
    :return: updated and compensated W, temp_means, temp_ages and conditions for update cond_horizontal,
    cond_vertical, cond_diagonal, cond_self for updating variance so not calc them again
    """
    x_overlap_idx = prev_grid_coords_x + np.sign(offset_x).astype(
        np.int32)  # figure out X direction of jumping, taking the overlaping center index in X
    cond_horizontal = (prev_grid_coords_y >= 0) & (prev_grid_coords_y < model_height) & (
            x_overlap_idx >= 0) & (x_overlap_idx < model_width)  # check if this crop is in image

    # horizontal overlapping
    temp_means = update_by_condition(cond_horizontal, temp_means, prev_means, W_H, x_grid_coords,
                                     y_grid_coords,
                                     x_overlap_idx, prev_grid_coords_y)
    temp_ages = update_by_condition(cond_horizontal, temp_ages, prev_ages, W_H, x_grid_coords, y_grid_coords,
                                    x_overlap_idx, prev_grid_coords_y)

    W = update_W_by_condition(W, W_H, x_grid_coords, y_grid_coords, cond_horizontal)
    # W[:, y_grid_coords[cond_horizontal], x_grid_coords[cond_horizontal]] += W_H[y_grid_coords[cond_horizontal],
    #                                                                             x_grid_coords[cond_horizontal]]

    # same for vertical
    y_overlap_idx = prev_grid_coords_y + np.sign(offset_y).astype(np.int32)
    cond_vertical = (y_overlap_idx >= 0) & (y_overlap_idx < model_height) & (prev_grid_coords_x >= 0) & \
                    (prev_grid_coords_x < model_width)

    temp_means = update_by_condition(cond_vertical, temp_means, prev_means, W_V, x_grid_coords, y_grid_coords,
                                     prev_grid_coords_x, y_overlap_idx)
    temp_ages = update_by_condition(cond_vertical, temp_ages, prev_ages, W_V, x_grid_coords, y_grid_coords,
                                    prev_grid_coords_x, y_overlap_idx)
    W = update_W_by_condition(W, W_V, x_grid_coords, y_grid_coords, cond_vertical)

    # same for diagonal
    x_overlap_idx = prev_grid_coords_x + np.sign(offset_x).astype(np.int32)
    y_overlap_idx = prev_grid_coords_y + np.sign(offset_y).astype(np.int32)
    cond_diagonal = (y_overlap_idx >= 0) & (y_overlap_idx < model_height) & (x_overlap_idx >= 0) & (
            x_overlap_idx < model_width)
    temp_means = update_by_condition(cond_diagonal, temp_means, prev_means, W_HV, x_grid_coords, y_grid_coords,
                                     x_overlap_idx, y_overlap_idx)
    temp_ages = update_by_condition(cond_diagonal, temp_ages, prev_ages, W_HV, x_grid_coords, y_grid_coords,
                                    x_overlap_idx, y_overlap_idx)
    W = update_W_by_condition(W, W_HV, x_grid_coords, y_grid_coords, cond_diagonal)

    # same for closest center
    cond_self = (prev_grid_coords_y >= 0) & (prev_grid_coords_y < model_height) & (prev_grid_coords_x >= 0) & (
            prev_grid_coords_x < model_width)
    temp_means = update_by_condition(cond_self, temp_means, prev_means, W_self, x_grid_coords, y_grid_coords,
                                     prev_grid_coords_x, prev_grid_coords_y)
    temp_ages = update_by_condition(cond_self, temp_ages, prev_ages, W_self, x_grid_coords, y_grid_coords,
                                    prev_grid_coords_x, prev_grid_coords_y)

    W = update_W_by_condition(W, W_self, x_grid_coords, y_grid_coords, cond_self)

    return W, temp_means, temp_ages, cond_horizontal, cond_vertical, cond_diagonal, cond_self


@jit(nopython=True)
def compensate_var(prev_vars, prev_means, means, W_H, W_V, W_HV, W_self, x_grid_coords, y_grid_coords,
                   prev_grid_coords_x, prev_grid_coords_y, offset_x, offset_y, cond_horizontal, cond_vertical,
                   cond_diagonal, cond_self):
    x_overlap_idx = prev_grid_coords_x + np.sign(offset_x).astype(
        np.int32)  # figure out X direction of jumping, taking the overlapping center index in X
    y_overlap_idx = prev_grid_coords_y + np.sign(offset_y).astype(np.int32)
    temp_var = np.zeros(prev_means.shape, dtype=prev_vars.dtype)

    temp_var = update_var_by_condition(cond_horizontal, temp_var, prev_vars, prev_means, means,
                                       W_H, x_grid_coords, y_grid_coords, x_overlap_idx,
                                       prev_grid_coords_y)

    temp_var = update_var_by_condition(cond_vertical, temp_var, prev_vars, prev_means, means,
                                       W_V, x_grid_coords, y_grid_coords, prev_grid_coords_x,
                                       y_overlap_idx)

    temp_var = update_var_by_condition(cond_diagonal, temp_var, prev_vars, prev_means, means, W_HV,
                                       x_grid_coords, y_grid_coords, x_overlap_idx,
                                       y_overlap_idx)

    temp_var = update_var_by_condition(cond_self, temp_var, prev_vars, prev_means, means, W_self,
                                       x_grid_coords, y_grid_coords, prev_grid_coords_x,
                                       prev_grid_coords_y)

    return temp_var


@jit(nopython=True)
def enlarge_pixels(input_array, b_size):
    rows = input_array.shape[0]
    cols = input_array.shape[1]
    enlarged_array = np.zeros((rows * b_size, cols * b_size), dtype=input_array.dtype)

    for i in range(b_size):
        for j in range(b_size):
            enlarged_array[i::b_size, j::b_size] = input_array

    return enlarged_array


@jit(nopython=True)
def convolve2d_with_padding(image, kernel):
    output = np.zeros_like(image)
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.zeros((image_height + 2 * pad_height, image_width + 2 * pad_width), dtype=image.dtype)
    padded_image[pad_height:-pad_height, pad_width:-pad_width] = image

    for y in prange(image_height):
        for x in prange(image_width):
            for i in prange(kernel_height):
                for j in prange(kernel_width):
                    output[y, x] += padded_image[y + i, x + j] * kernel[i, j]

    return output


@jit(nopython=True, parallel=True)
def get_chosen_means(means, model_index, jj, ii):
    model_index = model_index.flatten()
    num_elements = len(model_index)
    mns = np.zeros(num_elements, dtype=means.dtype)
    for i in prange(num_elements):
        mns[i] = means[model_index[i], jj[i], ii[i]]
    return mns


@jit(nopython=True, parallel=True)
def rebinMean(arr, factor):
    new_shape = (arr.shape[0] // factor[0], factor[0], arr.shape[1] // factor[1], factor[1])
    res = np.empty((new_shape[0], new_shape[2]), dtype=np.float32)

    for i in prange(new_shape[0]):
        for j in prange(new_shape[2]):
            total = 0.0
            for k in prange(factor[0]):
                for l in prange(factor[1]):
                    total += arr[i * factor[0] + k, j * factor[1] + l]
            res[i, j] = total / (factor[0] * factor[1])
    return res


@jit(nopython=True, parallel=True)
def update_vars_numba(means, ages, com_vars, alpha, gray_image: np.ndarray, models_to_update, model_index, h, w,
                      block_size, var_init, var_trim):
    jj, ii = np.arange(h * w) // w, np.arange(h * w) % w
    mns = get_chosen_means(means, model_index, jj, ii)
    mns = reshape_to_2d_array_numba(mns, (h, w))
    big_mean_index = enlarge_pixels(mns, block_size)  # extande the chosen models means upon the whole grid
    res = np.power(gray_image - big_mean_index, 2)
    maxes = rebinMax(res, (block_size, block_size))  # calc V for each grid for chosen model
    vars = com_vars * alpha + (1 - alpha) * maxes
    for k in prange(vars.shape[0]):
        for j in prange(vars.shape[1]):
            for i in prange(vars.shape[2]):
                if vars[k, j, i] < var_init and ages[k, j, i] == 0 and models_to_update[k, j, i]:
                    vars[k, j, i] = var_init
                if vars[k, j, i] < var_trim and models_to_update[k, j, i]:
                    vars[k, j, i] = var_trim
    return vars


@jit(nopython=True)
def update_means(com_means, alpha, cur_mean):
    """
    update the means according to eq (1)
    :param means: the object means
    :param com_means: compensated means
    :param alpha: the coefficient of mu in eq (1): a_com(t-1) / [a_com(t-1) + 1]
    :param cur_mean: the current mean as explained before
    """
    means = com_means * alpha + cur_mean * (1 - alpha)  # update mean
    return means


@jit(nopython=True)
def calc_probability(gray, det, temporal_property, spatial_property):
    neighborhood_size = (5, 5)
    kernel = np.ones(neighborhood_size) / (neighborhood_size[0] * neighborhood_size[1])
    alpha = 0.3

    temporal_property = alpha * temporal_property + (1 - alpha) * det / 255
    spatial_property = alpha * spatial_property + (1 - alpha) * \
                       convolve2d_with_padding(gray / 255, kernel)
    probs = temporal_property * spatial_property
    out = (probs * 255).astype(np.uint8)
    return out


@jit(nopython=True, parallel=True)
def suppression(gray, out, theta_d, big_mean, big_var):
    sqrt_theta_d = np.sqrt(theta_d)
    # big_std = np.sqrt(big_var)

    rows, cols = gray.shape
    threshold = np.zeros(gray.shape, dtype=big_var.dtype)

    for i in prange(rows):
        for j in prange(cols):
            threshold[i, j] = big_mean[i, j] + sqrt_theta_d * np.sqrt(big_var[i, j])

    for i in prange(rows):
        for j in prange(cols):
            if gray[i, j] < threshold[i, j]:
                out[i, j] = 0
    return out


# this function is the old suppression and currently not used
@jit(nopython=True, parallel=True)
def suppression_by_image(gray, out, theta_d):
    sqrt_theta_d = np.sqrt(theta_d)
    mn = np.mean(gray)
    std = np.std(gray)
    threshold = mn + sqrt_theta_d * std

    rows, cols = gray.shape
    for i in prange(rows):
        for j in prange(cols):
            if gray[i, j] < threshold:
                out[i, j] = 0
    return out


@jit(nopython=True, parallel=True)
def rebinMax(arr: np.ndarray, factor: tuple) -> np.ndarray:
    # identicle to rebin + max
    rows, cols = arr.shape[:2]
    res = np.zeros((rows // factor[0], cols // factor[1]), dtype=arr.dtype)

    for i in prange(0, rows // factor[0]):
        for j in prange(0, cols // factor[1]):
            max_val = arr[i * factor[0]:(i + 1) * factor[0], j * factor[1]:(j + 1) * factor[1]].max()
            res[i, j] = max_val
    return res


@jit(nopython=True)
def get_alpha(com_ages, models_to_update):
    """
    calc coefficient of the paper
    :param com_ages: compensated ages
    :param models_to_update:  the indexes of the chosen models to update
    :return: coefficient
    """
    rows, cols, depth = com_ages.shape  # Assuming com_ages and models_to_update have the same shape

    alpha = np.zeros_like(com_ages)

    for i in range(rows):
        for j in range(cols):
            for k in range(depth):
                alpha_val = com_ages[i, j, k] / (com_ages[i, j, k] + 1)

                if com_ages[i, j, k] < 1:
                    alpha_val = 0

                if not models_to_update[i, j, k]:
                    alpha_val = 1

                alpha[i, j, k] = alpha_val

    return alpha


@jit(nopython=True)
def calc_by_thresh(gray, big_means, big_vars, big_ages, theta):
    """
    decide each pixels are foreground by thresholding as in eq (16)
    :param gray: the image
    :param big_means: documented in other functions
    :param big_vars: documented in other functions
    :param big_ages: documented in other functions
    :param theta: the threshold
    :return: foreground-background matrix
    """
    rows, cols = gray.shape
    dist_img = np.power(gray - big_means, 2)
    out = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if big_ages[i, j] > 1 and dist_img[i, j] > theta * big_vars[i, j]:
                out[i, j] = 255
            else:
                out[i, j] = 0
    return out
