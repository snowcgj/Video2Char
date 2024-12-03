# -*- coding: utf-8 -*-
import cv2
import time
import os
import enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm
import imageio
from imageio import get_writer
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

# 这个是自己定义的配置文件
from config import *
from util import open_file_dialog as openPic

try:
    cached_font = ImageFont.truetype(ppdir + "/src/BRADHITC.TTF", 10)  # 字符缓存，使用正斜杠兼容不同操作系统
except OSError:
    cached_font = ImageFont.load_default()

# region 根据平台选择不同的命令
class Platform(enum.IntEnum):
    NT = 1
    POSIX = 2

if os.name == "nt":
    platform = Platform.NT
elif os.name == "posix":
    platform = Platform.POSIX
else:
    platform = Platform.POSIX

def clear_screen():
    if platform == Platform.NT:
        cmd = "cls"
    else:
        cmd = "clear"
    os.system(cmd)
# endregion

# region 配置的字符串以及长度
ascii_char = ASCII_CHAR
char_len = len(ascii_char)
# endregion

def show_image(img, timeout=0):
    cv2.namedWindow("image")
    cv2.imshow("image", img)
    cv2.waitKey(timeout * 1000)
    cv2.destroyAllWindows()

# 自动调整，也就是没输入什么大小，自动进行这个调整
def auto_thumb(img):
    height, width = img.shape[:2]
    if width >= D_WIDTH:
        height = height / (width / D_WIDTH) * 0.5
        width = D_WIDTH
    else:
        height = height / 2
    return cv2.resize(img, (int(width), int(height)))

# region 线程池处理帧，更快

def process_frame(frame):
    return to_chars_vectorized(frame, VIDEO_W_THUMB, VIDEO_H_THUMB)

def convert_char_frame_to_image(char_frame, font=cached_font, font_size=10):
    """
    将ASCII字符帧转换为PIL图像。
    """
    try:
        img = ascii_to_image(char_frame, font, font_size)
        return np.array(img)
    except Exception as e:
        print(f"转换帧时出错: {e}")
        return None

def save_gif_with_thread_pool(char_frames, output_gif, fps, font=cached_font, font_size=10):
    images = []

    # 定义线程池大小，根据您的CPU核心数进行调整
    max_workers = min(32, os.cpu_count() + 4)  # 这是一个常用的线程池大小策略

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(convert_char_frame_to_image, frame, font, font_size) for frame in char_frames]
        # 使用tqdm显示进度，保持顺序
        for future in tqdm(futures, total=len(char_frames), desc="正在保存为gif图："):
            img = future.result()  # 按提交顺序收集结果
            if img is not None:
                images.append(img)
    # 保存为GIF
    with get_writer(output_gif, fps=fps) as writer:
        for img in tqdm(images, desc="Saving GIF frames", total=len(images)):
            writer.append_data(img)
    print(f"ASCII视频已保存至 {output_gif}")

def save_mp4_with_cv2(char_frames, output_mp4, fps, font=cached_font, font_size=10):
    images = []
    # 定义线程池大小，根据您的CPU核心数进行调整
    max_workers = min(32, os.cpu_count() + 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(convert_char_frame_to_image, frame, font, font_size) for frame in char_frames]
        # 使用tqdm显示进度
        for future in tqdm(futures, total=len(char_frames), desc="正在保存为MP4："):
            img = future.result()  # 按提交顺序收集结果
            if img is not None:
                images.append(img)
    # 确保有帧可以写入
    if not images:
        print("没有可写入的视频帧。")
        return
    # 获取帧的尺寸
    height, width, layers = images[0].shape
    # 定义编码器，使用 'mp4v' 编码  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # fourcc 视频编码格式
    video_writer = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))
    with tqdm(total=len(images), desc="Saving MP4 frames") as tq:
        for img in images:
            # 将RGB转换为BGR，因为OpenCV使用BGR格式  （蓝、绿、红）
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(img_bgr)
            tq.update(1)

    video_writer.release()
    print(f"ASCII视频已保存至 {output_mp4}")

def video2char_async_threaded(file, progress_callback=None):
    '''
    异步处理逻辑核心函数
    '''
    video = cv2.VideoCapture(file)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"fps: {fps}, frame count: {frame_count}")
    char_frames = []
    
    if not video.isOpened():
        print("非支持的文件...")
        if progress_callback:
            progress_callback("错误: 文件无法打开")
        return
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for _ in tqdm(range(frame_count), desc="Processing frames: "):
            flag, frame = video.read()
            if not flag:
                break
            futures.append(executor.submit(process_frame, frame))
        
        if progress_callback:
            progress_callback("正在处理中...")
        
        for future in tqdm(futures, desc="正在加入到char_frames列表中"):
            char_frames.append(future.result())
    
    if progress_callback:
        progress_callback("处理中完成")
    
    return fps, char_frames

# 转化为字符串
def to_chars_vectorized(img, w_thumb=1.0, h_thumb=1.0):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if w_thumb == h_thumb == 1 and AUTO_THUMB:
        img_gray_resize = auto_thumb(img_gray)
    else:
        img_gray_resize = cv2.resize(img_gray, (int(img_gray.shape[1] / w_thumb), int(img_gray.shape[0] / h_thumb)))
    
    # 将灰度值映射到ASCII字符索引
    indices = (img_gray_resize / 256 * char_len).astype(int)
    indices[indices >= char_len] = char_len - 1  # 防止索引越界
    
    # 创建一个二维的字符数组
    chars_array = np.array(list(ascii_char))[indices]
    
    # 将二维字符数组转换为字符串
    chars = "\n".join("".join(row) for row in chars_array)
    return chars

def to_chars_print(img, w_thumb=1.0, h_thumb=1.0):
    # clear_screen()
    chars = to_chars_vectorized(img, w_thumb, h_thumb)
    return chars 

def ascii_to_image(ascii_str, font=cached_font, font_size=10):
    lines = ascii_str.split('\n')
    width = max(len(line) for line in lines)
    height = len(lines)
    img = Image.new('RGB', (width * font_size // 2, height * font_size), color='black')
    draw = ImageDraw.Draw(img)
    font = cached_font
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            draw.text((x * font_size // 2, y * font_size), char, font=font, fill=(255, 255, 255))
    return img

def main_process(file_path, ascii_set, save_option, output_path, w_thumb, h_thumb, progress_var):
    global ascii_char, char_len
    # 设置ASCII字符集
    ascii_char = ASCII_CHAR_SETS.get(ascii_set, ASCII_CHAR_SETS["dense"])
    char_len = len(ascii_char)
    
    img = cv2.imread(file_path)
    if img is not None:
        print("当前图片尺寸为宽：%s 高：%s" % (img.shape[1], img.shape[0]))
        chars = to_chars_print(img, w_thumb, h_thumb)
        # 保存到文件
        with open(output_path, "w+", encoding='utf-8') as f:
            f.write(chars)
        if save_option == "text":
            messagebox.showinfo("完成", f"ASCII图像已保存至 {output_path}")
        elif save_option == "image":
            img_ascii = ascii_to_image(chars)
            img_ascii.save(output_path)
            messagebox.showinfo("完成", f"ASCII图像已保存为图片至 {output_path}")
    else:
        print("文件非图片, 尝试打开为视频...")
        result = video2char_async_threaded(file_path)
        if result is None:
            messagebox.showerror("错误", "视频处理失败")
            return
        fps, char_frames = result
        # 根据保存选项保存
        if save_option == "gif":
            save_gif_with_thread_pool(char_frames, output_path, fps)
            messagebox.showinfo("完成", f"ASCII视频已保存至 {output_path}")
        elif save_option == "mp4":
            save_mp4_with_cv2(char_frames, output_path, fps)
            messagebox.showinfo("完成", f"ASCII视频已保存至 {output_path}")
        elif save_option == "play":
            for char_frame in char_frames:
                clear_screen()
                print(char_frame)
                time.sleep(1 / fps)
        else:
            save_gif_with_thread_pool(char_frames, output_path, fps)
            messagebox.showinfo("完成", f"ASCII视频已保存至 {output_path}")

# GUI 部分
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("视频和图片转ASCII工具")
        self.root.geometry("700x600")  # 增大窗口高度以适应所有组件

        # 创建一个主框架
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 使用grid布局管理器
        main_frame.columnconfigure(1, weight=1)  # 第二列（索引1）可拉伸

        # 文件选择
        tk.Label(main_frame, text="选择文件:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.file_path = tk.StringVar()
        self.file_entry = tk.Entry(main_frame, textvariable=self.file_path)
        self.file_entry.grid(row=0, column=1, sticky="ew", pady=5, padx=5)
        tk.Button(main_frame, text="浏览", command=self.browse_file).grid(row=0, column=2, sticky=tk.W, padx=5)

        # ASCII字符集选择
        tk.Label(main_frame, text="选择ASCII字符集:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.ascii_set = tk.StringVar(value="dense")
        ascii_options = ["dense", "light", "custom", "cc", "reverlight"]
        self.ascii_menu = tk.OptionMenu(main_frame, self.ascii_set, *ascii_options)
        self.ascii_menu.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)

        # 缩小比例
        tk.Label(main_frame, text="缩小比例:").grid(row=2, column=0, sticky=tk.W, pady=5)
        thumb_frame = tk.Frame(main_frame)
        thumb_frame.grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)
        tk.Label(thumb_frame, text="宽:").pack(side=tk.LEFT)
        self.w_thumb = tk.DoubleVar(value=1.0)
        tk.Entry(thumb_frame, textvariable=self.w_thumb, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(thumb_frame, text="高:").pack(side=tk.LEFT)
        self.h_thumb = tk.DoubleVar(value=1.0)
        tk.Entry(thumb_frame, textvariable=self.h_thumb, width=5).pack(side=tk.LEFT, padx=5)

        # 保存选项
        tk.Label(main_frame, text="保存选项:").grid(row=3, column=0, sticky=tk.NW, pady=5)
        self.save_option = tk.StringVar(value="gif")
        save_options = [("保存为GIF", "gif"), ("保存为MP4", "mp4"), ("保存为文本", "text"), ("保存为图片", "image"), ("播放", "play")]
        save_frame = tk.Frame(main_frame)
        save_frame.grid(row=3, column=1, sticky=tk.W, pady=5, padx=5)
        for text, mode in save_options:
            tk.Radiobutton(save_frame, text=text, variable=self.save_option, value=mode).pack(anchor=tk.W)

        # 输出路径
        tk.Label(main_frame, text="输出路径:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.output_path = tk.StringVar()
        self.output_entry = tk.Entry(main_frame, textvariable=self.output_path)
        self.output_entry.grid(row=4, column=1, sticky="ew", pady=5, padx=5)
        tk.Button(main_frame, text="浏览", command=self.browse_output).grid(row=4, column=2, sticky=tk.W, padx=5)

        # 进度条
        tk.Label(main_frame, text="进度:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, length=400, mode='indeterminate')
        self.progress.grid(row=5, column=1, columnspan=2, sticky="ew", pady=5, padx=5)

        # 开始按钮
        self.start_button = tk.Button(main_frame, text="开始转换", command=self.start_conversion)
        self.start_button.grid(row=6, column=0, columnspan=3, pady=20)

        # 调整窗口大小
        root.update_idletasks()
        self.root.minsize(700, 600)

    def browse_file(self):
        file_path = filedialog.askopenfilename(title="选择图片或视频文件", 
                                               filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp"), 
                                                          ("视频文件", "*.mp4;*.avi;*.mov;*.mkv")])
        if file_path:
            self.file_path.set(file_path)

    def browse_output(self):
        save_option = self.save_option.get()
        if save_option in ["gif", "mp4", "image"]:
            if save_option == "gif":
                filetypes = [("GIF 文件", "*.gif")]
                defaultextension = ".gif"
            elif save_option == "mp4":
                filetypes = [("MP4 文件", "*.mp4")]
                defaultextension = ".mp4"
            elif save_option == "image":
                filetypes = [("PNG 文件", "*.png"), ("JPEG 文件", "*.jpg")]
                defaultextension = ".png"
            file_path = filedialog.asksaveasfilename(defaultextension=defaultextension, 
                                                     filetypes=filetypes)
            if file_path:
                self.output_path.set(file_path)
        elif save_option == "text":
            file_path = filedialog.asksaveasfilename(defaultextension=".txt", 
                                                     filetypes=[("Text 文件", "*.txt")])
            if file_path:
                self.output_path.set(file_path)
        elif save_option == "play":
            self.output_path.set("")

    def start_conversion(self):
        file_path = self.file_path.get()
        if not file_path:
            messagebox.showerror("错误", "请先选择一个文件")
            return
        save_option = self.save_option.get()
        output_path = self.output_path.get()
        if save_option != "play" and not output_path:
            messagebox.showerror("错误", "请指定输出路径")
            return
        ascii_set = self.ascii_set.get()
        w_thumb = self.w_thumb.get()
        h_thumb = self.h_thumb.get()
        
        # 启动进度条
        self.progress.start()
        
        # 禁用按钮防止重复点击
        self.start_button.config(state=tk.DISABLED)
        
        # 运行在另一个线程中以避免阻塞GUI
        threading.Thread(target=self.run_process, args=(file_path, ascii_set, save_option, output_path, w_thumb, h_thumb)).start()

    def run_process(self, file_path, ascii_set, save_option, output_path, w_thumb, h_thumb):
        try:
            main_process(file_path, ascii_set, save_option, output_path, w_thumb, h_thumb, self.update_progress)
        except Exception as e:
            messagebox.showerror("错误", f"发生错误: {e}")
        finally:
            # 停止进度条
            self.progress.stop()
            # 启用按钮
            self.start_button.config(state=tk.NORMAL)

    def update_progress(self, message):
        # 这里可以根据需要更新GUI的进度显示
        print(message)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
