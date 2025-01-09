import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
class Student:
    def __init__(self):
        self.modified_file_path=None
        

    def creat_student(self):
                        # 加载Excel文件
        # 初始化Tkinter，然后隐藏主窗口
        Tk().withdraw()

        # 弹出文件选择器，让用户选择一个Excel文件
        file_path = askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])  # 只允许选择Excel文件

        # 检查用户是否真的选择了一个文件
        if not file_path:
            print("没有选择文件。")
            return

        # 检查是否存在student文件夹，如果不存在则创建
        if not os.path.exists('./student'):
            os.makedirs('./student')


        self.modified_file_path=file_path

    def get_students(self):
                # 指定Excel文件路径
        file_path=self.modified_file_path
        # 使用pandas读取Excel文件
        student_data = pd.read_excel(file_path)
        # 初始化一个空列表来存储学生信息
        students_list = []
        # 遍历DataFrame，为每个学生创建一个字典，包含姓名和处理状态
        for index, row in student_data.iterrows():
            student_info = {
                'name': row['姓名'],  # 假设Excel文件中有一个'姓名'列
                'status': row['处理状态'],  # 假设有一个'处理状态'列
                'class': row['班级'],  # 假设有一个'班级'列
                'donetimes': row['完成次数'],  # 假设有一个'班级'列
                'student_id': row['学号']  # 假设有一个'学号'列
            }
            students_list.append(student_info)
        return students_list,file_path

    def update_student_status_in_excel(self,student_id, new_status):
        # 读取Excel文件

        df = pd.read_excel(self.modified_file_path, engine='openpyxl')
        student_id=int(student_id)
        # 找到对应学号的行
        student_row = df['学号'] == student_id
        
        # 检查是否找到对应学号的学生
        if student_row.any():
            # 更新'处理状态'列
            df.loc[student_row, '处理状态'] = new_status
            # 保存修改后的DataFrame回同一个Excel文件
            df.to_excel(self.modified_file_path, index=False, engine='openpyxl')
            print("Status updated successfully.")
        else:
            print("Student ID not found.")