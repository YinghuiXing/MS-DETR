# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 15:48
# @Author  : WangSong
# @Email   : 1021707198@qq.com
# @File    : utils.py

import sys
import os

current_file_path = os.path.dirname(__file__)

sys.path.append(os.path.join(current_file_path, '..'))

from visualize import paintBBoxes