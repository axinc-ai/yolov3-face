import numpy as np
import cv2

def mapEquiangularX(x, W, cos):
    u = (x / W - 0.5) * cos + 0.5
    return u * W

def amapEquiangularX(u, W, cos):
    x = u + 0.5 * W * (cos - 1)
    return x / cos

def getEquiangularImage(image, rot, angle_range):
    """Transforming a image into an equiangular.

    Parameters
    ----------
    image : ndarray
        Transforming image.
    rot : float
        Horizontal angle[deg].
    angle_range : float
        Process within this range, centered on rot.
    Returns
    -------
    ndarray
        Transformed image

    """
    h, w, _ = image.shape
    map_x = np.zeros((h, w))
    map_y = np.zeros((h, w))
    step = angle_range / h
    for y in range(h):
        theta = (rot + (y - h * 0.5) * step) * np.pi / 180.0
        cos = np.abs(np.cos(theta))
        for x in range(w):
            map_x[y, x] = mapEquiangularX(x, w, cos)
            map_y[y, x] = y

    map_x = map_x.astype('float32')
    map_y = map_y.astype('float32')
    return cv2.remap( image, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)


def getEquiangularPoint(rot, angle_range, x, y, w, h):
    step = angle_range / h
    theta = (rot + (y - h * 0.5) * step) * np.pi / 180.0
    cos = np.abs(np.cos(theta))
    return amapEquiangularX(x, w, cos), y

def getEquiangularRect(rot, angle_range, x, y, w, h, imagew, imageh):
    """Transforming a rectangle into an equiangular.

    Parameters
    ----------
    rot : float
        Horizontal angle[deg].
    angle_range : float
        Process within this range, centered on rot.
    x, y, w, h : float
        x, y is left-top position. w, h is width and height

    Returns
    -------
    list
        Each transformed point.

    """
    result = []    
    xp = x
    yp = y
    result.append(getEquiangularPoint(rot, angle_range, xp, yp, imagew, imageh))
    xp = x + w
    yp = y
    result.append(getEquiangularPoint(rot, angle_range, xp, yp, imagew, imageh))
    xp = x
    yp = y + h
    result.append(getEquiangularPoint(rot, angle_range, xp, yp, imagew, imageh))
    xp = x + w
    yp = y + h
    result.append(getEquiangularPoint(rot, angle_range, xp, yp, imagew, imageh))
    return result
    
def getMinimumEnclosingRect(point_list):
    """Compute the rectangle containing the point list.

    Parameters
    ----------
    point_list : list
        point tuple.

    Returns
    -------
    Tuple
        rectangle

    """
    if len(point_list) == 0:
        return None, None, None, None

    x_list = [p[0] for p in point_list]
    y_list = [p[1] for p in point_list]

    return min(x_list), min(y_list), max(x_list), max(y_list)
