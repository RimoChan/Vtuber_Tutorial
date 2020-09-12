import contextlib
import time
import random


冷却时间 = 3

平均用时 = {}

@contextlib.contextmanager
def 计(名字):
    global 冷却时间
    开始时间 = time.time()
    yield
    用时 = time.time() - 开始时间
    if 名字 not in 平均用时: 
        平均用时[名字] = 用时
    平均用时[名字] = 0.9*平均用时[名字] + 0.1*用时
    冷却时间 -= 用时
    if 冷却时间 < 0:
        冷却时间 += 1
        for k, v in 平均用时.items():
            print(f'「{k}」帧率: %.2f' % (1/v))
