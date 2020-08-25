import sys
import time
import math

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

def 生成纹理(img):
    w, h = img.shape[:2]
    d = 2**int(max(math.log2(w), math.log2(h)) + 1)
    纹理 = np.zeros([d, d, 4], dtype=img.dtype)
    纹理[:w, :h] = img
    return 纹理, (w / d, h / d)

def opengl循环(所有图层, psd尺寸):
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
        glClearColor(1, 1, 1, 0)
        glClear(GL_COLOR_BUFFER_BIT)
        for 图层数据 in 所有图层:
            a, b, c, d = 图层数据['位置']
            z = 图层数据['深度']
            if type(z) in [int, float]:
                z1, z2, z3, z4 = [z, z, z, z]
            else:
                [z1, z2], [z3, z4] = z
            q, w = 图层数据['纹理座标']
            p1 = np.array([a, b, z1, 1, 0, 0, 0, 1])
            p2 = np.array([a, d, z2, 1, w, 0, 0, 1])
            p3 = np.array([c, d, z3, 1, w, q, 0, 1])
            p4 = np.array([c, b, z4, 1, 0, q, 0, 1])

            model = matrix.scale(2 / psd尺寸[0], 2 / psd尺寸[1], 1) @ \
                matrix.translate(-1, -1, 0) @ \
                matrix.rotate_ax(-math.pi / 2, axis=(0, 1))
            glBindTexture(GL_TEXTURE_2D, 图层数据['纹理编号'])
            glColor4f(1, 1, 1, 1)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glBegin(GL_QUADS)
            for p in [p1, p2, p3, p4]:
                a = p[:4]
                b = p[4:]
                a = a @ model
                z = a[2]
                a[0:2] *= z
                b *= z
                横旋转量 = 0
                if not 图层数据['名字'] == '身体':
                    横旋转量 = math.sin(time.time() * 5) / 30
                a = a @ matrix.translate(0, 0, -1) \
                      @ matrix.rotate_ax(横旋转量, axis=(0, 2)) \
                      @ matrix.translate(0, 0, 1)
                a = a @ matrix.perspective(999)
                glTexCoord4f(*b)
                glVertex4f(*a)
            glEnd()
        glfw.swap_buffers(window)


def 添加深度信息(所有图层):
    with open('深度2.yaml', encoding='utf8') as f:
        深度信息 = yaml.load(f)
    for 图层信息 in 所有图层:
        if 图层信息['名字'] in 深度信息:
            图层信息['深度'] = 深度信息[图层信息['名字']]


if __name__ == '__main__':
    psd = PSDImage.open('../res/莉沫酱过于简单版.psd')
    所有图层, size = 提取图层(psd)
    添加深度信息(所有图层)
    opengl循环(所有图层, size)
