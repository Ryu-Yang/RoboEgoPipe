import time

import rerun as rr
import numpy as np

class Viewer():
    def __init__(self):
        rr.init("RoboEgoPipe Viewer", spawn=True)
        
        # 设置时间轴
        rr.set_time("timestamp", timestamp=0)

    
    