import time
import logging
import threading

import cv2
import mediapipe as mp
import numpy as np


# 初始化 Mediapipe 的 Face Mesh 模块
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def 提取关键点(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    关键点 = []
    for face_landmarks in results.multi_face_landmarks:
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
            关键点.append(np.array([x, y], dtype=np.float32))
    
    return 关键点


def 生成构造点(关键点):
    def 中心(索引数组):
        return sum([关键点[i] for i in 索引数组]) / len(索引数组)

    眉心 = [9]
    下巴 = [152]
    鼻子 = [1]
    
    return 中心(眉心), 中心(下巴), 中心(鼻子)



def 生成特征(构造点):
    眉中心, 下巴中心, 鼻子中心 = 构造点
    中线 = 眉中心 - 下巴中心
    斜边 = 眉中心 - 鼻子中心
    横旋转量 = np.cross(中线, 斜边) / np.linalg.norm(中线)**2
    竖旋转量 = 中线 @ 斜边 / np.linalg.norm(中线)**2
    return np.array([横旋转量, 竖旋转量])


def 提取图片特征(img):
    关键点 = 提取关键点(img)
    if not 关键点:
        return None
    
    构造点 = 生成构造点(关键点)
    旋转量组 = 生成特征(构造点)
    return 旋转量组


def 捕捉循环():
    global 原点特征组
    global 特征组
    原点特征组 = 提取图片特征(cv2.imread('./res/std_face.jpg'))
    特征组 = 原点特征组 - 原点特征组
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    logging.warning('开始捕捉了！')
    while True:
        ret, img = cap.read()
        新特征组 = 提取图片特征(img)
        if 新特征组 is not None:
            特征组 = 新特征组 - 原点特征组
        time.sleep(1 / 60)


def 获取特征组():
    return 特征组


t = threading.Thread(target=捕捉循环)
t.setDaemon(True)
t.start()
logging.warning('捕捉线程启动中……')

if __name__ == '__main__':
    while True:
        time.sleep(0.1)
        print(获取特征组())
