import sys
import time
import math
import logging
import random

import numpy as np
import yaml

import glfw
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo

import psd_tools

import 现实

sys.path.append('../utils')
import matrix
from 截图 import *


Vtuber尺寸 = 512, 512


class 图层类:
    def __init__(self, 名字, bbox, z, npdata):
        self.名字 = 名字
        self.npdata = npdata
        self.纹理编号, 纹理座标 = self.生成opengl纹理()
        q, w = 纹理座标
        a, b, c, d = bbox
        if type(z) in [int, float]:
            深度 = np.array([[z, z], [z, z]])
        else:
            深度 = np.array(z)
        assert len(深度.shape) == 2
        self.shape = 深度.shape

        [[p1, p2],
         [p4, p3]] = np.array([
             [[a, b, 0, 1, 0, 0, 0, 1], [a, d, 0, 1, w, 0, 0, 1]],
             [[c, b, 0, 1, 0, q, 0, 1], [c, d, 0, 1, w, q, 0, 1]],
         ])
        x, y = self.shape
        self.顶点组 = np.zeros(shape=[x, y, 8])
        for i in range(x):
            for j in range(y):
                self.顶点组[i, j] = p1 + (p4-p1)*i/(x-1) + (p2-p1)*j/(y-1)
                self.顶点组[i, j, 2] = 深度[i, j]

    def 生成opengl纹理(self):
        w, h = self.npdata.shape[:2]
        d = 2**int(max(math.log2(w), math.log2(h)) + 1)
        纹理 = np.zeros([d, d, 4], dtype=self.npdata.dtype)
        纹理[:w, :h] = self.npdata
        纹理座标 = (w / d, h / d)

        width, height = 纹理.shape[:2]
        纹理编号 = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, 纹理编号)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_FLOAT, 纹理)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)

        return 纹理编号, 纹理座标

    def 方块组导出(self):
        源 = self.顶点组
        x, y, _ = 源.shape
        方块组 = []
        for i in range(x-1):
            for j in range(y-1):
                方块组.append(
                    [[源[i, j], 源[i, j+1]],
                     [源[i+1, j], 源[i+1, j+1]]]
                )
        return 方块组


class vtuber:
    def __init__(self, psd路径, yaml路径='信息.yaml'):
        psd = psd_tools.PSDImage.open(psd路径)
        with open(yaml路径, encoding='utf8') as f:
            信息 = yaml.safe_load(f)
        self.所有图层 = []
        self.psd尺寸 = psd.size
        def dfs(图层, path=''):
            if 图层.is_group():
                for i in 图层:
                    dfs(i, path + 图层.name + '/')
            else:
                a, b, c, d = 图层.bbox
                npdata = 图层.numpy()
                npdata[:, :, 0], npdata[:, :, 2] = npdata[:, :, 2].copy(), npdata[:, :, 0].copy()
                名字 = path+图层.name
                if 名字 not in 信息:
                    logging.warning(f'图层「{名字}」找不到信息，丢了！')
                    return
                self.所有图层.append(图层类(
                    名字=名字,
                    z=信息[名字]['深度'],
                    bbox=(b, a, d, c),
                    npdata=npdata
                ))
        for 图层 in psd:
            dfs(图层)

    def opengl绘图循环(self, window, 数据源, log=True, line_box=False):
        def draw(图层):
            所有顶点 = []
            for 方块 in 图层.方块组导出():
                [[p1, p2],
                 [p4, p3]] = 方块
                所有顶点 += [p1, p2, p3, p4]
            ps = np.array(所有顶点)

            model = \
                matrix.scale(2 / self.psd尺寸[0], 2 / self.psd尺寸[1], 1) @ \
                matrix.translate(-1, -1, 0) @ \
                matrix.rotate_ax(-math.pi / 2, axis=(0, 1))

            view = \
                matrix.translate(0, 0, -1) @ \
                matrix.rotate_ax(横旋转量, axis=(0, 2)) @ \
                matrix.rotate_ax(竖旋转量, axis=(2, 1)) @ \
                matrix.translate(0, 0, 1)

            a = ps[:, :4]
            b = ps[:, 4:]
            a = a @ model
            z = a[:, 2:3]
            a[:, :2] *= z
            b *= z
            if 图层.名字 != '身体':
                a = a @ view
            a = a @ matrix.perspective(999)
            for i in range(len(所有顶点)):
                if i % 4 == 0:
                    glBegin(GL_QUADS)
                glTexCoord4f(*b[i])
                glVertex4f(*a[i])
                if i % 4 == 3:
                    glEnd()

        平均用时 = None
        while not glfw.window_should_close(window):
            渲染开始时间 = time.time()
            glfw.poll_events()
            glClearColor(0, 0, 0, 0)
            glClear(GL_COLOR_BUFFER_BIT)
            横旋转量, 竖旋转量 = 数据源()
            for 图层 in self.所有图层:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, 图层.纹理编号)
                glColor4f(1, 1, 1, 1)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                draw(图层)
                if line_box:
                    glDisable(GL_TEXTURE_2D)
                    glColor4f(1, 1, 1, 0.3)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                    draw(图层)
            glfw.swap_buffers(window)
            用时 = time.time() - 渲染开始时间
            if 平均用时 is None:
                平均用时 = 用时
            平均用时 = 0.9*平均用时 + 0.1*用时
            if random.random() < 平均用时 and log:
                print('帧率: %.2f' % (1/平均用时))


缓冲特征 = None
def 特征缓冲(缓冲比例=0.95):
    global 缓冲特征
    新特征 = 现实.获取特征组()
    if 缓冲特征 is None:
        缓冲特征 = 新特征
    else:
        缓冲特征 = 缓冲特征 * 缓冲比例 + 新特征 * (1 - 缓冲比例)
    return 缓冲特征


def init_window():
    def 超融合():
        glfw.window_hint(glfw.DECORATED, False)
        glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, True)
        glfw.window_hint(glfw.FLOATING, True)
    glfw.init()
    超融合()
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.RESIZABLE, False)
    window = glfw.create_window(*Vtuber尺寸, 'Vtuber', None, None)
    glfw.make_context_current(window)
    monitor_size = glfw.get_video_mode(glfw.get_primary_monitor()).size
    glfw.set_window_pos(window, monitor_size.width - Vtuber尺寸[0], monitor_size.height - Vtuber尺寸[1])
    glViewport(0, 0, *Vtuber尺寸)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glEnable(GL_MULTISAMPLE)
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
    return window



if __name__ == '__main__':
    现实.启动()

    window = init_window()

    莉沫酱 = vtuber('../res/莉沫酱较简单版.psd')
    莉沫酱.opengl绘图循环(window, 数据源=特征缓冲)
