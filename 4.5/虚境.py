import math
import yaml
import numpy as np
import glfw
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from psd_tools import PSDImage
from rimo_utils import matrix
import 现实



class 图层:
    def __init__(self, psd):
        self.所有图层, self.psd尺寸 = self.提取图层(psd)
        self.添加深度信息()

    def 提取图层(self, psd):
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

    def 添加深度信息(self):
        with open('./4.5/深度.yaml', encoding='utf8') as f:
            深度信息 = yaml.load(f, Loader=yaml.FullLoader)
        for 图层信息 in self.所有图层:
            if 图层信息['名字'] in 深度信息:
                图层信息['深度'] = 深度信息[图层信息['名字']]


class OpenGL渲染:
    def __init__(self, 所有图层, psd尺寸):
        self.所有图层 = 所有图层
        self.psd尺寸 = psd尺寸
        self.缓冲特征 = None

    def 特征缓冲(self):
        缓冲比例 = 0.8
        新特征 = 现实.获取特征组()
        if self.缓冲特征 is None:
            self.缓冲特征 = 新特征
        else:
            self.缓冲特征 = self.缓冲特征 * 缓冲比例 + 新特征 * (1 - 缓冲比例)
        return self.缓冲特征

    def 超融合(self):
        glfw.window_hint(glfw.DECORATED, False)
        glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, True)
        glfw.window_hint(glfw.FLOATING, True)

    def 生成纹理(self, img):
        w, h = img.shape[:2]
        d = 2**int(max(math.log2(w), math.log2(h)) + 1)
        纹理 = np.zeros([d, d, 4], dtype=img.dtype)
        纹理[:w, :h] = img
        return 纹理, (w / d, h / d)

    def opengl绘图循环(self):
        Vtuber尺寸 = 512, 512
        
        glfw.init()
        self.超融合()
        glfw.window_hint(glfw.RESIZABLE, False)
        window = glfw.create_window(*Vtuber尺寸, 'Vtuber', None, None)
        glfw.make_context_current(window)
        monitor_size = glfw.get_video_mode(glfw.get_primary_monitor()).size
        glfw.set_window_pos(window, monitor_size.width - Vtuber尺寸[0], monitor_size.height - Vtuber尺寸[1])

        glViewport(0, 0, *Vtuber尺寸)

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)

        for 图层数据 in self.所有图层:
            纹理编号 = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, 纹理编号)
            纹理, 纹理座标 = self.生成纹理(图层数据['npdata'])
            width, height = 纹理.shape[:2]
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_FLOAT, 纹理)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glGenerateMipmap(GL_TEXTURE_2D)
            图层数据['纹理编号'] = 纹理编号
            图层数据['纹理座标'] = 纹理座标

        while not glfw.window_should_close(window):
            glfw.poll_events()
            glClearColor(0, 0, 0, 0)
            glClear(GL_COLOR_BUFFER_BIT)
            横旋转量, 竖旋转量 = self.特征缓冲()
            for 图层数据 in self.所有图层:
                a, b, c, d = 图层数据['位置']
                z = 图层数据['深度']
                if type(z) in [int, float]:
                    z1, z2, z3, z4 = [z, z, z, z]
                else:
                    [z1, z2], [z3, z4] = z
                q, w = 图层数据['纹理座标']
                p1 = np.array([a, b, z1, 1, 0, 0, 0, z1])
                p2 = np.array([a, d, z2, 1, z2 * w, 0, 0, z2])
                p3 = np.array([c, d, z3, 1, z3 * w, z3 * q, 0, z3])
                p4 = np.array([c, b, z4, 1, 0, z4 * q, 0, z4])

                model = matrix.scale(2 / self.psd尺寸[0], 2 / self.psd尺寸[1], 1) @ \
                    matrix.translate(-1, -1, 0) @ \
                    matrix.rotate_ax(-math.pi / 2, axis=(0, 1))
                glBindTexture(GL_TEXTURE_2D, 图层数据['纹理编号'])
                glColor4f(1, 1, 1, 1)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glBegin(GL_QUADS)
                for p in [p1, p2, p3, p4]:
                    a = p[:4]
                    b = p[4:8]
                    a = a @ model
                    a[0:2] *= a[2]
                    if not 图层数据['名字'][:2] == '身体':
                        a = a @ matrix.translate(0, 0, -1) \
                              @ matrix.rotate_ax(横旋转量, axis=(0, 2)) \
                              @ matrix.rotate_ax(竖旋转量, axis=(2, 1)) \
                              @ matrix.translate(0, 0, 1)
                    a = a @ matrix.perspective(999)
                    glTexCoord4f(*b)
                    glVertex4f(*a)
                glEnd()
            glfw.swap_buffers(window)


if __name__ == '__main__':
    psd = PSDImage.open('./res/莉沫酱过于简单版.psd')
    图层实例 = 图层(psd)
    渲染实例 = OpenGL渲染(图层实例.所有图层, 图层实例.psd尺寸)
    渲染实例.opengl绘图循环()
