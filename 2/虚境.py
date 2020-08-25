import sys
import math

import cv2
import numpy as np
import yaml

import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

from psd_tools import PSDImage

sys.path.append('../utils')
import matrix
from 截图 import *


def 提取图层(psd):
    所有图层 = []
    def dfs(图层, path=''):
        if 图层.is_group():
            for i in 图层:
                dfs(i, path + 图层.name + '/')
        else:
            a, b, c, d = 图层.bbox
            npdata = 图层.numpy()
            npdata[:, :, 0], npdata[:, :, 2] = npdata[:, :, 2].copy(), npdata[:, :, 0].copy()
            所有图层.append({'名字': path + 图层.name, '位置': (b, a, d, c), 'npdata': npdata})
    for 图层 in psd:
        dfs(图层)
    return 所有图层, psd.size


def 测试图层叠加(所有图层):
    img = np.ones([1024, 1024, 4], dtype=np.float32)
    for 图层数据 in 所有图层:
        a, b, c, d = 图层数据['位置']
        新图层 = 图层数据['npdata']
        alpha = 新图层[:, :, 3]
        for i in range(3):
            img[a:c, b:d, i] = img[a:c, b:d, i] * (1 - alpha) + 新图层[:, :, i] * alpha
    cv2.imshow('', img)
    cv2.imwrite('1.jpg', (img*255).astype(np.uint8))
    cv2.waitKey()


def opengl绘图循环(所有图层, psd尺寸):
    Vtuber尺寸 = 512, 512

    glfw.init()
    glfw.window_hint(glfw.RESIZABLE, False)
    window = glfw.create_window(*Vtuber尺寸, 'Vtuber', None, None)
    glfw.make_context_current(window)
    glViewport(0, 0, *Vtuber尺寸)

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    for 图层数据 in 所有图层:
        纹理编号 = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, 纹理编号)
        纹理 = cv2.resize(图层数据['npdata'], (1024, 1024))
        width, height = 纹理.shape[:2]
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_FLOAT, 纹理)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)
        图层数据['纹理编号'] = 纹理编号

    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClearColor(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        for 图层数据 in 所有图层:
            a, b, c, d = 图层数据['位置']
            p1 = np.array([a, b, 0, 1, 1, 0])
            p2 = np.array([a, d, 0, 1, 1, 1])
            p3 = np.array([c, d, 0, 1, 0, 1])
            p4 = np.array([c, b, 0, 1, 0, 0])
            model = matrix.scale(2 / psd尺寸[0], 2 / psd尺寸[1], 1) @ \
                matrix.translate(-1, -1, 0) @ \
                matrix.rotate_ax(-math.pi / 2, axis=(0, 1))
            glBindTexture(GL_TEXTURE_2D, 图层数据['纹理编号'])
            glColor4f(1, 1, 1, 1)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glBegin(GL_QUADS)
            for p in [p1, p2, p3, p4]:
                a = p[:4]
                b = p[4:6]
                a = a @ model
                glVertex4f(*a)
                glTexCoord2f(*b)
            glEnd()
        glfw.swap_buffers(window)


def opengl清楚绘图循环(所有图层, psd尺寸):
    def 生成纹理(img):
        w, h = img.shape[:2]
        d = 2**int(max(math.log2(w), math.log2(h)) + 1)
        纹理 = np.zeros([d, d, 4], dtype=img.dtype)
        纹理[:w, :h] = img
        return 纹理, (w / d, h / d)

    Vtuber尺寸 = 512, 512

    glfw.init()
    glfw.window_hint(glfw.RESIZABLE, False)
    window = glfw.create_window(*Vtuber尺寸, 'Vtuber', None, None)
    glfw.make_context_current(window)
    glViewport(0, 0, *Vtuber尺寸)

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    for 图层数据 in 所有图层:
        纹理编号 = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, 纹理编号)
        纹理, 纹理座标 = 生成纹理(图层数据['npdata'])
        width, height = 纹理.shape[:2]
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_FLOAT, 纹理)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)
        图层数据['纹理编号'] = 纹理编号
        图层数据['纹理座标'] = 纹理座标

    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClearColor(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        for 图层数据 in 所有图层:
            a, b, c, d = 图层数据['位置']
            q, w = 图层数据['纹理座标']
            p1 = np.array([a, b, 0, 1, 0, 0])
            p2 = np.array([a, d, 0, 1, w, 0])
            p3 = np.array([c, d, 0, 1, w, q])
            p4 = np.array([c, b, 0, 1, 0, q])
            model = matrix.scale(2 / psd尺寸[0], 2 / psd尺寸[1], 1) @ \
                matrix.translate(-1, -1, 0) @ \
                matrix.rotate_ax(-math.pi / 2, axis=(0, 1))
            glBindTexture(GL_TEXTURE_2D, 图层数据['纹理编号'])
            glColor4f(1, 1, 1, 1)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glBegin(GL_QUADS)
            for p in [p1, p2, p3, p4]:
                a = p[:4]
                b = p[4:6]
                a = a @ model
                glTexCoord2f(*b)
                glVertex4f(*a)
            glEnd()
        glfw.swap_buffers(window)


if __name__ == '__main__':
    psd = PSDImage.open('../res/莉沫酱过于简单版.psd')
    所有图层, size = 提取图层(psd)
    # 测试图层叠加(所有图层)
    opengl清楚绘图循环(所有图层, size)
