import numpy as np


def no_overflow(i, addition, limit):
    if i + addition >= limit:
        return limit - 1
    else:
        return i + addition - 1


def get_grid_coordinates(grid_size, img_size):
    coords = []
    patch_size_y = img_size[0] // grid_size
    patch_size_x = img_size[1] // grid_size
    for y in range(0, img_size[0], patch_size_y):
        for x in range(0, img_size[1], patch_size_x):
            x0 = x
            y0 = y
            x1 = no_overflow(x, patch_size_x, img_size[1])
            y1 = no_overflow(y, patch_size_y, img_size[0])
            coords.append([x0, y0, x1, y1])
    return coords


def get_prev_centers(mat, centers):
    return [np.matmul(mat, center) for center in centers]


def correct_overlap(i1, i2):
    a = abs(i1 - i2)
    if a < 1:
        return 1 - a
    return 0


def overlap(coords1, coords2):  # returns 0 if rectangles don't intersect
    dx = min(coords1[2], coords2[2]) - max(coords1[0], coords2[0]) + correct_overlap(coords1[2], coords2[2]) + \
         correct_overlap(coords1[0], coords2[0])
    dy = min(coords1[3], coords2[3]) - max(coords1[1], coords2[1]) + correct_overlap(coords1[3], coords2[3]) + \
         correct_overlap(coords1[1], coords2[1])
    if (dx >= 0) and (dy >= 0):
        return (dx + 1) * (dy + 1)
    return 0


def rect_area(x1, y1, x2, y2):
    xdiff = abs(x1 - x2)  # Using absolute value to ignore negatives
    ydiff = abs(y1 - y2)

    return xdiff * ydiff
