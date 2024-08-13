import cv2
import mediapipe as mp
import numpy as np

def 生成构造点(关键点):
    def 中心(索引数组):
        return sum([关键点[i] for i in 索引数组]) / len(索引数组)

    眉心 = [9]
    下巴 = [152]
    鼻子 = [1]
    
    return 中心(眉心), 中心(下巴), 中心(鼻子)

# 初始化 Mediapipe 的 Face Mesh 模块
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

# 初始化绘图模块
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像从 BGR 转换为 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理图像，检测面部关键点
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 提取所有关键点坐标
            关键点 = []
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                关键点.append(np.array([x, y]))

            # 计算构造点
            if 关键点:
                构造点 = 生成构造点(关键点)

                # 将构造点连接成多边形并填充淡蓝色
                构造点_array = np.array(构造点, dtype=np.int32)

                # 设置半透明效果
                alpha = 0.5  # 半透明度
                overlay = frame.copy()
                cv2.fillPoly(overlay, [构造点_array], color=(255, 200, 200))
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # 绘制构造点
                for point in 构造点:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

            # 绘制所有面部关键点
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))
    
    # 显示图像
    cv2.imshow('构造点和关键点', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
