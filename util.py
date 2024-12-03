import easygui

def open_file_dialog():
    file_path = easygui.fileopenbox("选择图片", "选择一个文件", filetypes=["*.png", "*.jpg", "*.jpeg", "*.bmp"])
    return file_path

def study1():
    flag = input("是否手动缩小比例(Y/n)?:") 
    if flag.lower() == "y":
        print("这就是默认值的写法")
# file_path = open_file_dialog()

