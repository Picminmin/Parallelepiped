import numpy as np
from hyperrectangle import HyperRectangle

class Parallelepiped_manhattan():
    def __init__(self, category_num):
        self.category_num = category_num
        self.hr_list = []
