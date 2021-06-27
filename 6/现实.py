import time
import logging
import threading
import multiprocessing

import cv2
import dlib
import numpy as np
from rimo_utils import 计时


detector = dlib.get_frontal_face_detector()


def 多边形面积(a):
    a = np.array(a)
    x = a[:, 0]
    y = a[:, 1]
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


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
    return np.array(关键点)


def 计算旋转量(关键点):
    def 中心(索引数组):
        return sum([关键点[i] for i in 索引数组]) / len(索引数组)
    左眉 = [18, 19, 20, 21]
    右眉 = [22, 23, 24, 25]
    下巴 = [6, 7, 8, 9, 10]
    鼻子 = [29, 30]
    眉中心, 下巴中心, 鼻子中心 = 中心(左眉 + 右眉), 中心(下巴), 中心(鼻子)
    中线 = 眉中心 - 下巴中心
    斜边 = 眉中心 - 鼻子中心
    中线长 = np.linalg.norm(中线)
    横旋转量 = np.cross(中线, 斜边) / 中线长**2
    竖旋转量 = 中线 @ 斜边 / 中线长**2
    Z旋转量 = np.cross(中线, [0, 1]) / 中线长
    return np.array([横旋转量, 竖旋转量, Z旋转量])


def 计算嘴大小(关键点):
    边缘 = 关键点[0:17]
    嘴边缘 = 关键点[48:60]
    嘴大小 = 多边形面积(嘴边缘) / 多边形面积(边缘)
    return np.array([嘴大小])


def 计算相对位置(img, 脸位置):
    x = (脸位置.top() + 脸位置.bottom())/2/img.shape[0]
    y = (脸位置.left() + 脸位置.right())/2/img.shape[1]
    y = 1 - y
    相对位置 = np.array([x, y])
    return 相对位置


def 计算脸大小(关键点):
    边缘 = 关键点[0:17]
    t = 多边形面积(边缘)**0.5
    return np.array([t])


def 计算眼睛大小(关键点):
    边缘 = 关键点[0:17]
    左 = 多边形面积(关键点[36:42]) / 多边形面积(边缘)
    右 = 多边形面积(关键点[42:48]) / 多边形面积(边缘)
    return np.array([左, 右])


def 计算眉毛高度(关键点): 
    边缘 = 关键点[0:17]
    左 = 多边形面积([*关键点[18:22]]+[关键点[38], 关键点[37]]) / 多边形面积(边缘)
    右 = 多边形面积([*关键点[22:26]]+[关键点[44], 关键点[43]]) / 多边形面积(边缘)
    return np.array([左, 右])


def 提取图片特征(img):
    脸位置 = 人脸定位(img)
    if not 脸位置:
        return None
    相对位置 = 计算相对位置(img, 脸位置)
    关键点 = 提取关键点(img, 脸位置)
    旋转量组 = 计算旋转量(关键点)
    脸大小 = 计算脸大小(关键点)
    眼睛大小 = 计算眼睛大小(关键点)
    嘴大小 = 计算嘴大小(关键点)
    眉毛高度 = 计算眉毛高度(关键点)
    
    img //= 2
    img[脸位置.top():脸位置.bottom(), 脸位置.left():脸位置.right()] *= 2 
    for i, (px, py) in enumerate(关键点):
        cv2.putText(img, str(i), (int(px), int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255))
    
    return np.concatenate([旋转量组, 相对位置, 嘴大小, 脸大小, 眼睛大小, 眉毛高度])


原点特征组 = 提取图片特征(cv2.imread('../res/std_face.jpg'))
特征组 = 原点特征组 - 原点特征组


def 捕捉循环(pipe):
    global 原点特征组
    global 特征组
    cap = cv2.VideoCapture(0)
    logging.warning('开始捕捉了！')
    while True:
        with 计时.帧率计('提特征'):
            ret, img = cap.read()
            新特征组 = 提取图片特征(img)
            cv2.imshow('', img[:, ::-1])
            cv2.waitKey(1)
            if 新特征组 is not None:
                特征组 = 新特征组 - 原点特征组
            pipe.send(特征组)


def 获取特征组():
    global 特征组
    return 特征组


def 转移():
    global 特征组
    logging.warning('转移线程启动了！')
    while True:
        特征组 = pipe[1].recv()


pipe = multiprocessing.Pipe()


def 启动():
    t = threading.Thread(target=转移)
    t.setDaemon(True)
    t.start()
    logging.warning('捕捉进程启动中……')
    p = multiprocessing.Process(target=捕捉循环, args=(pipe[0],))
    p.daemon = True
    p.start()


if __name__ == '__main__':
    启动()
    np.set_printoptions(precision=3, suppress=True)
    while True:
        time.sleep(0.1)
        # print(获取特征组())
