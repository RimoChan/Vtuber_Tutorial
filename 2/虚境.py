import time
import math
import os

import cv2
import numpy as np
import yaml

import glfw
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *

from psd_tools import PSDImage


import matrix


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
    img = np.ones([1000, 1000, 4], dtype=np.float32)
    for 图层数据 in 所有图层:
        a, b, c, d = 图层数据['位置']
        新图层 = 图层数据['npdata']
        alpha = 新图层[:, :, 3]
        for i in range(3):
            img[a:c, b:d, i] = img[a:c, b:d, i] * (1 - alpha) + 新图层[:, :, i] * alpha
    cv2.imshow('', img)
    
    cv2.imwrite('1.jpg',(img*255).astype(np.uint8))
    cv2.waitKey()


def opengl绘图循环(所有图层, psd尺寸):
    Vtuber尺寸 = 500, 500

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
        for 图层数据 in 所有图层:
            a, b, c, d = 图层数据['位置']
            z = 图层数据['深度']
            p1 = np.array([a, b, z, 1,  1, 0])
            p2 = np.array([a, d, z, 1,  1, 1])
            p3 = np.array([c, d, z, 1,  0, 1])
            p4 = np.array([c, b, z, 1,  0, 0])
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
                a[0] *= a[2]
                a[1] *= a[2]
                if not 图层数据['名字'][:2] == '身体':
                    横旋转量 = math.sin(time.clock() * 5) / 30
                    竖旋转量 = 0
                    a = a @ \
                        matrix.translate(0, 0, -1) @ \
                        matrix.rotate_ax(横旋转量, axis=(0, 2)) @ \
                        matrix.rotate_ax(竖旋转量, axis=(2, 1)) @ \
                        matrix.translate(0, 0, 1)
                a = a @ matrix.perspective(999)
                glVertex4f(*a)
                glTexCoord2f(*b)
            glEnd()
        glfw.swap_buffers(window)


def opengl截图():
    glReadBuffer(GL_FRONT)
    h = w = 500
    data = glReadPixels(0, 0, h, w, GL_RGBA, GL_UNSIGNED_BYTE)
    img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4)).copy()
    img = img[:, :, :3]
    img = img[::-1, :, ::-1]
    return img


def 添加深度信息(所有图层):
    with open('深度.yaml', encoding='utf8') as f:
        深度信息 = yaml.load(f)
    for 图层信息 in 所有图层:
        if 图层信息['名字'] in 深度信息:
            图层信息['深度'] = 深度信息[图层信息['名字']]


if __name__ == '__main__':
    psd = PSDImage.open('../res/rimo.psd')
    所有图层, size = 提取图层(psd)
    # 测试图层叠加(所有图层)
    添加深度信息(所有图层)
    opengl绘图循环(所有图层, size)
