import logging
import threading
import multiprocessing
import time

import cv2
import numpy as np
import dlib


detector = dlib.get_frontal_face_detector()


def 人脸定位(img):
    dets = detector(img, 0)
    if not dets:
        return None
    return max(dets, key=lambda det: (det.right() - det.left()) * (det.bottom() - det.top()))


predictor = dlib.shape_predictor('../res/shape_predictor_68_face_landmarks.dat')


def 提取关键点(img, 脸位置):
    landmark_shape = predictor(img, 脸位置)
    关键点 = []
    for i in range(68):
        pos = landmark_shape.part(i)
        关键点.append(np.array([pos.x, pos.y], dtype=np.float32))
    return 关键点


def 生成构造点(关键点):
    def 中心(索引数组):
        return sum([关键点[i] for i in 索引数组]) / len(索引数组)
    左眉 = [18, 19, 20, 21]
    右眉 = [22, 23, 24, 25]
    下巴 = [6, 7, 8, 9, 10]
    鼻子 = [29, 30]
    return 中心(左眉 + 右眉), 中心(下巴), 中心(鼻子)


def 生成特征(构造点):
    眉中心, 下巴中心, 鼻子中心 = 构造点
    中线 = 眉中心 - 下巴中心
    斜边 = 眉中心 - 鼻子中心
    横旋转量 = np.cross(中线, 斜边) / np.linalg.norm(中线)**2
    竖旋转量 = 中线 @ 斜边 / np.linalg.norm(中线)**2
    return np.array([横旋转量, 竖旋转量])


def 提取图片特征(img):
    脸位置 = 人脸定位(img)
    if not 脸位置:
        return None
    关键点 = 提取关键点(img, 脸位置)
    构造点 = 生成构造点(关键点)
    旋转量组 = 生成特征(构造点)
    return 旋转量组


原点特征组 = 提取图片特征(cv2.imread('../res/std_face.jpg'))
特征组 = 原点特征组 - 原点特征组    

def 捕捉循环(pipe):
    global 原点特征组
    global 特征组
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    logging.warning('开始捕捉了！')
    while True:
        ret, img = cap.read()
        新特征组 = 提取图片特征(img)
        if 新特征组 is not None:
            特征组 = 新特征组 - 原点特征组
        time.sleep(0.01)
        pipe.send(特征组)


def 获取特征组():
    global 特征组
    return 特征组

def 转移(): 
    global 特征组
    while True:
        特征组 = pipe[1].recv()

pipe = multiprocessing.Pipe()
def 启动():
    logging.warning('捕捉进程启动中……')
    p = multiprocessing.Process(target=捕捉循环, args=(pipe[0],))
    p.start()


t = threading.Thread(target=转移)
t.setDaemon(True)
t.start()
logging.warning('捕捉线程启动中……')


if __name__ == '__main__':
    启动()
    while True:
        time.sleep(0.1)
        print(获取特征组())
