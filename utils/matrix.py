import math
from math import sin, cos
import functools


import numpy as np
np.set_printoptions(suppress=True)


@functools.lru_cache(maxsize=128)
def scale(x, y, z):
    a = np.eye(4, dtype=np.float32)
    a[0, 0] = x
    a[1, 1] = y
    a[2, 2] = z
    return a


@functools.lru_cache(maxsize=128)
def rotate_ax(r, axis: tuple):
    a = np.eye(4, dtype=np.float32)
    a[axis[0], axis[0]] = cos(r)
    a[axis[0], axis[1]] = sin(r)
    a[axis[1], axis[0]] = -sin(r)
    a[axis[1], axis[1]] = cos(r)
    return a


@functools.lru_cache(maxsize=128)
def rotate(angle, v):
    v = np.array(v, dtype=np.float32)
    v /= np.linalg.norm(v)
    a = np.array([0, 0, 1])
    b = np.cross(v, a)
    if np.linalg.norm(b) < np.linalg.norm(v) * np.linalg.norm(a) / 100:
        new_a = np.array([0, 1, 0])
        b = np.cross(a, new_a)
    c = np.cross(v, b)

    rm = np.array([b, c, v])
    arm = np.linalg.inv(rm)
    
    rm4 = np.eye(4, dtype=np.float32)
    rm4[:3,:3] = rm
    arm4 = np.eye(4, dtype=np.float32)
    arm4[:3,:3] = arm
    
    return arm4 @ rotate_ax(angle, axis=(0, 1)) @ rm4
    

@functools.lru_cache(maxsize=128)
def translate(x, y, z):
    a = np.eye(4, dtype=np.float32)
    a[3, 0] = x
    a[3, 1] = y
    a[3, 2] = z
    return a


@functools.lru_cache(maxsize=128)
def perspective(view_distance):
    a = np.eye(4, dtype=np.float32)
    a[2, 2] = 1 / view_distance
    a[3, 2] = -0.0001
    a[2, 3] = 1
    a[3, 3] = 0
    return a


def test():
    a = np.eye(4, dtype=np.float32)
    a[3, 3] = 2
    return a
