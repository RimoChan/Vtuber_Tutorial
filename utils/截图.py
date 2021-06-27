import time
import os

import numpy as np
import cv2
import imageio
from OpenGL.GL import *
from OpenGL.GLU import *


def opengl截图(size, 反转颜色=True):
    glReadBuffer(GL_FRONT)
    h, w = size
    data = glReadPixels(0, 0, h, w, GL_RGBA, GL_UNSIGNED_BYTE)
    img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4)).copy()
    if 反转颜色:
        img[:, :, :3] = img[:, :, :3][:, :, ::-1]
    img = img[::-1, :, :]
    return img


def opengl截图一闪(size):
    cv2.imwrite('x.jpg', opengl截图(size))
    exit()

开始时间 = None
图组 = []
def opengl连续截图(size, 时间, 缓冲=0):
    global 开始时间
    global 图组
    now = time.time()
    if 开始时间 is None:
        开始时间 = now
    if now-开始时间 > 时间+缓冲:
        图组 = [cv2.cvtColor(图, cv2.COLOR_BGR2RGB) for 图 in 图组]
        图组 = 图组[::5]
        print(f'fps: {len(图组)/时间}')
        imageio.mimsave("test.gif", 图组, fps=len(图组)/时间)
        os.system('gif2webp test.gif -lossy -min_size -m 6 -mt -o test.webp')
        exit()
    if now-开始时间>缓冲: 
        图组.append(opengl截图(size))
