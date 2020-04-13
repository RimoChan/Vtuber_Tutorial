import threading
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


predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
def 提取关键点(img, 脸位置):
    landmark_shape = predictor(img, 脸位置)
    关键点 = []
    for i in range(68):
        pos = landmark_shape.part(i)
        关键点.append(np.array([pos.x, pos.y], dtype=np.float32))
    return 关键点


def 生成构造点(关键点):
    左眉 = [19, 20, 21]
    右眉 = [22, 23, 24]
    下巴 = [6, 7, 8, 9, 10]
    鼻子 = [29, 30]

    眉中心 = sum([关键点[i] for i in 左眉 + 右眉]) / 6
    下巴中心 = sum([关键点[i] for i in 下巴]) / 5
    鼻子中心 = sum([关键点[i] for i in 鼻子]) / 2

    return 眉中心, 下巴中心, 鼻子中心


def 生成特征(构造点):
    眉中心, 下巴中心, 鼻子中心 = 构造点
    中线 = 眉中心 - 下巴中心
    斜边 = 眉中心 - 鼻子中心
    横旋转量 = np.cross(中线, 斜边) / np.linalg.norm(中线)**2
    竖旋转量 = 中线 @ 斜边 / np.linalg.norm(中线)**2 - 0.325
    return 横旋转量, 竖旋转量


特征组 = np.array([0, 0])
def 捕捉循环(): 
    global 特征组
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        脸位置 = 人脸定位(img)
        if not 脸位置:
            continue
        关键点 = 提取关键点(img, 脸位置)
        构造点 = 生成构造点(关键点)
        横旋转量, 竖旋转量 = 生成特征(构造点)
        特征组 = np.array([横旋转量, 竖旋转量])
        cv2.imshow('', img)
        cv2.waitKey(1)
        time.sleep(1/60)


def 获取特征组():
    return 特征组


t = threading.Thread(target=捕捉循环)
t.setDaemon(True)
t.start()
