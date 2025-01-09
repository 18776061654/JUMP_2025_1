from PySide6.QtWidgets import (QTableWidget, QTableWidgetItem, QMessageBox, 
                              QFileDialog, QProgressDialog)
from PySide6.QtCore import Qt, Signal
import pandas as pd
import os
import shutil
import json
import traceback

class StudentManager:
    def __init__(self, main_window):
        self.main_window = main_window
        self.current_student = None
        self.results_folder = './results'
        self.setup_tables()
        
        # 连接导入按钮的信号
        if hasattr(self.main_window, 'import_button'):  # 如果有导入按钮
            self.main_window.import_button.clicked.connect(self.import_students)
    
    def setup_tables(self):
        """设置三个表格的列标题"""
        # 修改表头，只显示学号、姓名和完成次数
        headers = ['学号', '姓名', '完成次数']
        tables = [
            self.main_window.studentList_1,  # 待测评
            self.main_window.studentList_2,  # 已测评
            self.main_window.studentList_3   # 测评中
        ]
        
        for table in tables:
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.setSelectionBehavior(QTableWidget.SelectRows)
            # 设置列宽
            table.setColumnWidth(0, 120)  # 学号列
            table.setColumnWidth(1, 80)   # 姓名列
            table.setColumnWidth(2, 80)   # 完成次数/进度列
    
    def import_students(self):
        """导入学生信息"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "选择Excel文件",
            "",
            "Excel files (*.xlsx *.xls)"
        )
        
        if not file_path:
            return
            
        try:
            # 读取Excel文件
            df = pd.read_excel(file_path)
            required_columns = ['student_id', 'name', 'gender', 'class', 'test_count', 'status']
            
            # 验证文件格式
            if not all(col in df.columns for col in required_columns):
                QMessageBox.warning(self.main_window, "错误", "Excel文件格式不正确")
                return
                
            # 清空现有表格
            for table in [self.main_window.studentList_1, 
                         self.main_window.studentList_2,
                         self.main_window.studentList_3]:
                table.setRowCount(0)
            
            # 分类添加学生
            for _, row in df.iterrows():
                self.add_student_to_table(row)
                
            # 使用新的批量创建方法
            self.create_student_folders_batch(df['student_id'].tolist())
            
            QMessageBox.information(self.main_window, "成功", "学生信息导入成功！")
                
        except Exception as e:
            QMessageBox.critical(self.main_window, "错误", f"导入失败: {str(e)}")
            print(f"导入错误详情: {str(e)}")  # 添加详细错误输出
    
    def add_student_to_table(self, student_data):
        """将学生添加到对应表格"""
        # 确定目标表格
        if student_data['status'] == '未测评':
            table = self.main_window.studentList_1
        elif student_data['status'] == '已测评':
            table = self.main_window.studentList_2
        else:  # 测评中
            table = self.main_window.studentList_3
            
        row = table.rowCount()
        table.setRowCount(row + 1)
        
        # 只设置学号、姓名和完成次数
        items = [
            student_data['student_id'],
            student_data['name'],
            str(student_data['test_count'])
        ]
        
        for col, item_text in enumerate(items):
            item = QTableWidgetItem(str(item_text))
            item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, col, item)
    
    def create_student_folders(self, create_new=False):
        """为学生创建结果文件夹"""
        student_id = self.main_window.sno_label.text()
        if not student_id:
            QMessageBox.warning(self.main_window, "提示", "请先选择学生")
            return None, None

        # 创建学生主文件夹
        student_folder = os.path.join(self.results_folder, str(student_id))
        os.makedirs(student_folder, exist_ok=True)

        if create_new:
            # 使用当前选中的测试次数
            test_count = self.main_window.current_test_number
            test_folder = os.path.join(student_folder, f'test_{test_count}')
            speed_folder = os.path.join(test_folder, 'speed')
            jump_folder = os.path.join(test_folder, 'jump')
            
            # 检查视频文件是否存在
            speed_video = os.path.join(speed_folder, "SpeedVideo.avi")
            jump_video = os.path.join(jump_folder, "JumpVideo.avi")
            
            if os.path.exists(speed_video) or os.path.exists(jump_video):
                reply = QMessageBox.question(
                    self.main_window,
                    "确认",
                    f"测试{test_count}已有录制视频，是否覆盖？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return None, None
                
                # 如果选择覆盖，删除已存在的视频文件
                try:
                    if os.path.exists(speed_video):
                        os.remove(speed_video)
                    if os.path.exists(jump_video):
                        os.remove(jump_video)
                except Exception as e:
                    print(f"删除旧视频文件时出错: {str(e)}")
                    return None, None
        else:
            # 使用当前选中的测试次数
            current_test = self.main_window.current_test_number
            test_folder = os.path.join(student_folder, f'test_{current_test}')
            speed_folder = os.path.join(test_folder, 'speed')
            jump_folder = os.path.join(test_folder, 'jump')

        # 创建必要的文件夹
        os.makedirs(speed_folder, exist_ok=True)
        os.makedirs(jump_folder, exist_ok=True)

        return speed_folder, jump_folder

    def get_next_test_number(self, student_folder):
        """获取下一个测试次数"""
        if not os.path.exists(student_folder):
            return 1
        
        # 获取所有test_开头的文件夹
        test_folders = [d for d in os.listdir(student_folder) 
                       if os.path.isdir(os.path.join(student_folder, d)) 
                       and d.startswith('test_')]
        
        if not test_folders:
            return 1
        
        # 检查已有的测试次数
        existing_numbers = set(int(folder.split('_')[1]) for folder in test_folders)
        
        # 从1开始找第一个可用的测试次数
        for i in range(1, 4):  # 最多3次测试
            if i not in existing_numbers:
                return i
        
        # 如果1-3都被使用了，返回3（使用最后一个测试文件夹）
        return 3

    def create_student_folders_batch(self, student_ids):
        """批量创建学生文件夹（用于导入时）"""
        for student_id in student_ids:
            # 只创建学生的主文件夹
            student_folder = os.path.join(self.results_folder, str(student_id))
            if not os.path.exists(student_folder):
                os.makedirs(student_folder)
                # 不再自动创建测试文件夹
                # for i in range(1, 4):
                #     test_folder = os.path.join(student_folder, f'test_{i}')
                #     os.makedirs(test_folder)
    
    def update_student_status(self, student_id, status, progress=None):
        """更新学生状态
        Args:
            student_id: 学生ID
            status: 状态（"未测评"/"测评中"/"已测评"）
            progress: 可选的进度值（0-100）
        """
        try:
            if student_id:
                # 更新Excel中的状态
                excel_file = os.path.join('student', 'student.xlsx')
                if os.path.exists(excel_file):
                    df = pd.read_excel(excel_file)
                    df['student_id'] = df['student_id'].astype(str)
                    student_id = str(student_id)
                    
                    mask = df['student_id'] == student_id
                    if any(mask):
                        df.loc[mask, 'status'] = status
                        if progress is not None:
                            df.loc[mask, 'test_count'] = f"{progress}%"
                        df.to_excel(excel_file, index=False)
                        
                        # 更新显示
                        self.update_student_display()
                        
                        # 打印状态更新信息
                        if progress is not None:
                            print(f"学生 {student_id} 状态更新为: {status} ({progress}%)")
                        else:
                            print(f"学生 {student_id} 状态更新为: {status}")
        except Exception as e:
            print(f"更新学生状态时出错: {str(e)}")
            traceback.print_exc()
    
    def find_student(self, student_id):
        """在所有表格中查找学生信息"""
        for table in [self.main_window.studentList_1,
                     self.main_window.studentList_2,
                     self.main_window.studentList_3]:
            for row in range(table.rowCount()):
                if table.item(row, 0).text() == str(student_id):
                    return {
                        'student_id': student_id,
                        'name': table.item(row, 1).text(),
                        'test_count': table.item(row, 2).text(),
                        'status': self.get_table_status(table),
                        'gender': '',  # 保持这些字段为空，因为不显示
                        'class': ''
                    }
        return None
    
    def get_table_status(self, table):
        """根据表格确定状态"""
        if table == self.main_window.studentList_1:
            return "未测评"
        elif table == self.main_window.studentList_2:
            return "已测评"
        else:
            return "测评中"
    
    def remove_from_all_tables(self, student_id):
        """从所有表格中移除学生"""
        for table in [self.main_window.studentList_1,
                     self.main_window.studentList_2,
                     self.main_window.studentList_3]:
            for row in range(table.rowCount()):
                if table.item(row, 0).text() == str(student_id):
                    table.removeRow(row)
                    break
    
    def update_progress(self, progress_value, student_id):
        """更新理进度"""
        # 不打印每个进度更新
        self.update_student_status(student_id, "测评中", progress_value)
    
    def load_student_scores(self):
        """加载学生成绩信息"""
        student_id = self.main_window.sno_label.text()
        if student_id:
            student_folder = os.path.join(self.results_folder, student_id)
            current_test = self.main_window.current_test_number
            test_folder = os.path.join(student_folder, f'test_{current_test}')
            
            try:
                # 尝试加载分数文件
                score_file = os.path.join(test_folder, 'jump', 'score_jump.json')
                if os.path.exists(score_file):
                    with open(score_file, 'r') as f:
                        scores = json.load(f)
                        # 更新UI显示
                        self.main_window.labelTakeOffScore.setText(f"{scores.get('take_off_score', 'N/A')}")
                        self.main_window.labelHipExtensionScore.setText(f"{scores.get('hip_extension_score', 'N/A')}")
                        self.main_window.labelAbdominalContractionScore.setText(f"{scores.get('abdominal_contraction_score', 'N/A')}")
                        self.main_window.labelAllScore.setText(f"{scores.get('all_score', 'N/A')}")
                else:
                    # 如果文件不存在，显示N/A
                    self.main_window.labelTakeOffScore.setText("N/A")
                    self.main_window.labelHipExtensionScore.setText("N/A")
                    self.main_window.labelAbdominalContractionScore.setText("N/A")
                    self.main_window.labelAllScore.setText("N/A")
            except Exception as e:
                print(f"加载学生分数时出错: {str(e)}")

    def update_student_display(self):
        """更新学生列表显示，重新导入文件"""
        try:
            # 直接从现有的Excel文件读取
            excel_file = os.path.join('student', 'student.xlsx')
            if os.path.exists(excel_file):
                df = pd.read_excel(excel_file)
                
                # 清空现有表格
                for table in [self.main_window.studentList_1, 
                             self.main_window.studentList_2, 
                             self.main_window.studentList_3]:
                    table.clearContents()
                    table.setRowCount(0)
                
                # 重新填充表格
                for _, row in df.iterrows():
                    self.add_student_to_table(row)
        except Exception as e:
            print(f"更新学生显示时出错: {str(e)}")
            traceback.print_exc()