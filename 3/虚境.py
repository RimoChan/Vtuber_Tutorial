import time
import math
import random
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

import 现实


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
    return 所有图层


def 添加深度信息(所有图层):
    with open('深度.yaml', encoding='utf8') as f:
        深度信息 = yaml.load(f)
    for 图层信息 in 所有图层:
        if 图层信息['名字'] in 深度信息:
            图层信息['深度'] = 深度信息[图层信息['名字']]


缓冲特征 = np.array([0, 0])
缓冲动量 = np.array([0, 0])
def 特征缓冲():
    global 缓冲特征
    global 缓冲动量
    缓冲比例 = 0.99
    动量比例 = 0.9
    新特征 = 现实.获取特征组()
    缓冲动量 = 缓冲动量 * 动量比例 + (新特征 - 缓冲特征) * (1 - 动量比例)
    if np.linalg.norm(缓冲动量) < 0.02:
        缓冲比例 += 0.009 * (0.02 - np.linalg.norm(缓冲动量)) / 0.02
    缓冲特征 = 缓冲特征 * 缓冲比例 + 新特征 * (1 - 缓冲比例)
    return 缓冲特征


def opengl绘图循环(所有图层):
    glfw.init()
    glfw.window_hint(glfw.RESIZABLE, False)
    window = glfw.create_window(500, 500, 'Vtuber', None, None)
    glfw.make_context_current(window)
    glViewport(0, 0, 500, 500)

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    for 图层数据 in 所有图层:
        纹理编号 = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, 纹理编号)
        纹理 = cv2.resize(图层数据['npdata'], (512, 512))
        width, height = 纹理.shape[:2]
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_FLOAT, 纹理)
        glGenerateMipmap(GL_TEXTURE_2D)
        图层数据['纹理编号'] = 纹理编号

    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClearColor(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT)
        for 图层数据 in 所有图层:
            a, b, c, d = 图层数据['位置']
            z = 图层数据['深度']
            p1 = np.array([a, b, z, 1,  0, 0])
            p2 = np.array([a, d, z, 1,  1, 0])
            p3 = np.array([c, d, z, 1,  1, 1])
            p4 = np.array([c, b, z, 1,  0, 1])
            model = matrix.scale(1 / 250, 1 / 250, 1) @ \
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
                    横角, 竖角 = 特征缓冲()
                    a = a @ \
                        matrix.translate(0, 0, -1) @ \
                        matrix.rotate_ax(横角, axis=(0, 2)) @ \
                        matrix.rotate_ax(竖角, axis=(2, 1)) @ \
                        matrix.translate(0, 0, 1)
                a = a @ matrix.perspective(999)
                glTexCoord2f(*b)
                glVertex4f(*a)
            glEnd()
        glfw.swap_buffers(window)


if __name__ == '__main__':
    psd = PSDImage.open('rimo.psd')
    所有图层 = 提取图层(psd)
    添加深度信息(所有图层)
    opengl绘图循环(所有图层)
