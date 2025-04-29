import os

# 假设项目文件夹为当前工作区，你可以根据实际情况修改
project_folder = '.'  

def check_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            if "写入成功" in content:
                print(f"文件 {file_path} 包含 '写入成功' 字样。")
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")

def traverse_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            check_file(file_path)

# 遍历项目文件夹
traverse_folder(project_folder)