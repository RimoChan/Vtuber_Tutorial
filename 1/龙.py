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
    旋转量 = np.cross(中线, 斜边) / np.linalg.norm(中线)**2
    return 旋转量


def 画图(旋转量):
    img = np.ones([512, 512], dtype=np.float32)
    脸长 = 200
    中心 = 256, 256
    左眼 = int(220 - 旋转量 * 脸长), 249
    右眼 = int(292 - 旋转量 * 脸长), 249
    嘴 = int(256 - 旋转量 * 脸长 * 0.5), 310
    cv2.circle(img, 中心, 100, 0, 1)
    cv2.circle(img, 左眼, 15, 0, 1)
    cv2.circle(img, 右眼, 15, 0, 1)
    cv2.circle(img, 嘴, 5, 0, 1)
    return img


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        脸位置 = 人脸定位(img)
        if not 脸位置:
            cv2.imshow('self', img)
            cv2.waitKey(1)
            continue
        关键点 = 提取关键点(img, 脸位置)
        # for i, (px, py) in enumerate(关键点):
        #     cv2.putText(img, str(i), (int(px),int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255))
        构造点 = 生成构造点(关键点)
        # for i, (px, py) in enumerate(构造点):
        #     cv2.putText(img, str(i), (int(px),int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255))
        旋转量 = 生成特征(构造点)
        # cv2.putText(img, '%.3f' % 旋转量,
        #             (int(构造点[-1][0]), int(构造点[-1][1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
        cv2.imshow('self', img)
        cv2.imshow('Vtuber', 画图(旋转量))
        cv2.waitKey(1)
