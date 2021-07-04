import time
import threading

import numpy as np
import pyvirtualcam


def start(vtuber, size):
    r, c = size
    def q():
        with pyvirtualcam.Camera(width=r, height=c, fps=30) as cam:
            base = np.zeros(shape=(c, r, 3), dtype=np.uint8)
            while True:
                img = vtuber.获取截图(False)
                base[:, (r-c)//2:(r-c)//2+c] = img[:, :, :3]
                cam.send(base)
                time.sleep(0.01)
    t = threading.Thread(target=q)
    t.setDaemon(True)
    t.start()
