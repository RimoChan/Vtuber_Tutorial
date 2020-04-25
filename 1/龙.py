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


def 画图(横旋转量, 竖旋转量):
    img = np.ones([512, 512], dtype=np.float32)
    脸长 = 200
    中心 = 256, 256
    左眼 = int(220 + 横旋转量 * 脸长), int(249 + 竖旋转量 * 脸长)
    右眼 = int(292 + 横旋转量 * 脸长), int(249 + 竖旋转量 * 脸长)
    嘴 = int(256 + 横旋转量 * 脸长 / 2), int(310 + 竖旋转量 * 脸长 / 2)
    cv2.circle(img, 中心, 100, 0, 1)
    cv2.circle(img, 左眼, 15, 0, 1)
    cv2.circle(img, 右眼, 15, 0, 1)
    cv2.circle(img, 嘴, 5, 0, 1)
    return img


def 提取图片特征(img):
    脸位置 = 人脸定位(img)
    if not 脸位置:
        cv2.imshow('self', img)
        cv2.waitKey(1)
        return None
    关键点 = 提取关键点(img, 脸位置)
    # for i, (px, py) in enumerate(关键点):
    #     cv2.putText(img, str(i), (int(px),int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255))
    构造点 = 生成构造点(关键点)
    # for i, (px, py) in enumerate(构造点):
    #     cv2.putText(img, str(i), (int(px),int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255))
    旋转量组 = 生成特征(构造点)
    # cv2.putText(img, '%.3f' % 旋转量,
    #             (int(构造点[-1][0]), int(构造点[-1][1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
    cv2.imshow('self', img)
    return 旋转量组


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    原点特征组 = 提取图片特征(cv2.imread('../res/std_face.jpg'))
    特征组 = 原点特征组 - 原点特征组
    while True:
        ret, img = cap.read()
        # img = cv2.flip(img, 1)
        新特征组 = 提取图片特征(img)
        if 新特征组 is not None:
            特征组 = 新特征组 - 原点特征组
        横旋转量, 竖旋转量 = 特征组
        cv2.imshow('Vtuber', 画图(横旋转量, 竖旋转量))
        cv2.waitKey(1)
