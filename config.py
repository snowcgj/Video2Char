# -*- coding: utf-8 -*-
# Date : 2019-02-10 15:12:27
import os

# 由密到疏字符组合
ASCII_CHAR = r"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
ASCII_CHAR_SETS = {
    "dense": r"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
    "light": r" .:-=+*#%@",
    "custom": r"█▓▒░ ",
    "cc":r"█▓▒░▁▂▃▄▅▆▇",
    "reverlight":r"%@#*+=-:. "
}


# 自动缩小至默认宽度
AUTO_THUMB = True
# 默认宽度
D_WIDTH = 140

# 视频缩小比例
VIDEO_W_THUMB = 9
VIDEO_H_THUMB = 18

# 清屏间隔,根据视频FPS调整
VIDEO_FLASH_TIME = 1 / 100

# 获得一个路径，
pdir= os.path.realpath(__file__) # E:\Github\vhar\config.py
ppdir=os.path.dirname(pdir)
fontpath=ppdir+"\src\BRADHITC.TTF"  # 测试下字体路径  E:\Github\video2char\src\BRADHITC.TTF
print(ppdir)
print(fontpath)
