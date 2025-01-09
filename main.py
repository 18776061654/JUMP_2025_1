import datetime
import logging
import cv2
import sys
import shutil
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QInputDialog, QListWidget, QFileDialog, QDialog, QVBoxLayout, QListWidgetItem,QTableWidget,QTableWidgetItem, QMessageBox,QWidget
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtCore import QTimer, Qt, QTime, QThread, Signal
from PySide6.QtUiTools import QUiLoader
import os
from collections import namedtuple
from queue import Queue
import json
from posture.JumpAnalyzer import JumpAnalyzer
from posture.PoseMatcher import PoseMatcher
from posture.key_score import KeyScoreCalculator
# 在创建 QApplication 对象之前设置全局属性
QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
# 引入自定义模块
from main_part.ui_handlers import setup_ui, connect_signals
from main_part.student_module import StudentManager
from main_part.utils import get_video_file_paths,convert_cv_qt
# from main_part.initposture import initposture_MainWindow
from main_part.camera_handler import CameraRecorder,CameraController
from speed.qt_speed import Speed_VideoProcessor
from main_part.show_results import ShowResults
from student.show_stu_score import StudentInfoTable
from student.student import Student
from main_part.image_viewer_with_mark import ImageWindow
from main_part.utils import VideoDialog, CalibrationDialog
import pandas as pd
import traceback
import warnings
from main_part.pose_display_thread import PoseDisplayThread
import numpy as np
from main_part.weight_assign import WeightApp
# 在程序开始时添加
warnings.filterwarnings("ignore", message="The predicted simcc values are normalized for visualization")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.create_folder()  # 创建结果文件夹
        self.setup_ui()  # 设置界面
        self.initialize_variables()  # 初始化变量
        self.setup_timers()  # 设置计时器
        self.setup_cameras()  # 初始化相机

        # 添加当前测试次数属性
        self.current_test_number = 1

        # 初始化 StudentManager
        self.student_manager = StudentManager(self)

        self.connect_signals()  # 连接信号与槽
        self.setup_student_manager()

        self.resize(1578, 888)

        self.jump_analyzer = JumpAnalyzer()  # 初始化JumpAnalyzer
        self.pose_matcher = PoseMatcher()    # 初始化PoseMatcher

        self.current_display_student_id = None  # 添加当前显示学生ID的属性
        self.pose_display_thread = None  # 添加显示线程的属性

    def loadstudents(self):
        # 修改这里，调用正确的方法名
        self.student_manager.import_students()
    def setup_ui(self):
        """ 加载UI文件并初始化基本布局 """
        setup_ui(self)  # 从 ui_handlers 导入
        


    def initialize_variables(self):
        """ 类变量和状态标志 """
        self.speed_camera_id = 'video=PC Camera'
        self.jump_camera_id = 'video=Camera'
        self.speed_file = None
        self.jump_file = None
        self.elapsed_time = 0
        self.timer_started = False
        self.cout = 0
        self.is_jump_video_processing = False
        self.is_speed_video_processing = False
        self.current_display = "camera1"
        self.recorder1_open = True
        self.recorder2_open = True



        self.student = Student()


    def setup_timers(self):
        """ 设置计时器用于更新界面显示 """
        # 初始化计时器显示为 00:00:00
        try:
            self.timerLabel.setText("00:00:00")
            
            # 创建并设置更新计时器
            self.update_timer = QTimer()
            self.update_timer.timeout.connect(self.update_timer_label)

            # 创建并设置图像更新计时器
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_images)
        except Exception as e:
            print(f"设置计时器时出错: {str(e)}")
            traceback.print_exc()

    def setup_cameras(self):
        """ 初始化相机控制器 """
        self.camera1 = CameraController(0)  # 第一个摄像头
        self.camera2 = CameraController(1)  # 第二个摄像头
        self.recorder1 = CameraRecorder()  # 摄像头1的录像器
        self.recorder2 = CameraRecorder()  # 摄像头2的录像器

    def connect_signals(self):
        """ 自动连接信号与槽 """
        connect_signals(self)  # 从 ui_handlers 导入

    def set_photo_image(self):
        pixmap = QPixmap("public/data/user.png")  # 替换为实际图片路径
        self.Ca_photo.setPixmap(pixmap)
        self.Ca_photo.setScaledContents(True)


    def load_standardBut_xx(self):
       try:
            self.weight_window = WeightApp()
            self.weight_window.show()
       except Exception as e:
           print(f"打开 WeightApp 时出错: {str(e)}")

    def show_calibration_dialog(self):
        self.calibration_dialog = CalibrationDialog(self)
        self.calibration_dialog.video_source_button.clicked.connect(self.setBoard_video_source)
        self.calibration_dialog.camera_button.clicked.connect(self.setBoard_camera_source)
        self.calibration_dialog.exec_()

    def show_video_preview_image(self):
        """显示速度和跳远视频预览的缩略图"""
        try:
            if not hasattr(self, 'sele_person_file') or not self.current_test_number:
                return

            # 构建视频文件路径
            speed_folder = os.path.join(self.sele_person_file, f'test_{self.current_test_number}', 'speed')
            jump_folder = os.path.join(self.sele_person_file, f'test_{self.current_test_number}', 'jump')
            speed_video = os.path.join(speed_folder, "SpeedVideo.avi")
            jump_video = os.path.join(jump_folder, "JumpVideo.avi")

            # 清除按钮图标和文字
            self.Ca_run_but.setIcon(QIcon())
            self.Ca_run_but.setText("")
            self.Ca_jump_but.setIcon(QIcon())
            self.Ca_jump_but.setText("")

            # 显示速度视频预览
            if os.path.exists(speed_video):
                cap = cv2.VideoCapture(speed_video)
                ret, frame = cap.read()
                if ret:
                    image = convert_cv_qt(frame, self.Ca_run_but.width(), self.Ca_run_but.height())
                    self.Ca_run_but.setIcon(QIcon(QPixmap.fromImage(image)))
                    self.Ca_run_but.setIconSize(self.Ca_run_but.size())
                cap.release()
            else:
                self.Ca_run_but.setText("无速度视频")

            # 显示跳远视频预览
            if os.path.exists(jump_video):
                cap = cv2.VideoCapture(jump_video)
                ret, frame = cap.read()
                if ret:
                    image = convert_cv_qt(frame, self.Ca_jump_but.width(), self.Ca_jump_but.height())
                    self.Ca_jump_but.setIcon(QIcon(QPixmap.fromImage(image)))
                    self.Ca_jump_but.setIconSize(self.Ca_jump_but.size())
                cap.release()
            else:
                self.Ca_jump_but.setText("无跳远视频")

        except Exception as e:
            print(f"更新视频预览时出错: {str(e)}")
            traceback.print_exc()

    def show_runvideo_preview(self):
        speed_file_name, jump_file_name=get_video_file_paths(self.student_manager)

        self.video_dialog = VideoDialog(speed_file_name, self)
        self.video_dialog.show()

    def show_jumpvideo_preview(self):
        speed_file_name, jump_file_name=get_video_file_paths(self.student_manager)

        self.video_dialog = VideoDialog(jump_file_name, self)
        self.video_dialog.show()


    def load_speed_video(self):
        """加载速度视频"""
        if not hasattr(self, 'sele_person_file') or not self.sele_person_file:
            QMessageBox.warning(self, "提示", "请先选择学生")
            return
        
        # 使用当前测试次数获取文件夹路径
        student_folder = self.sele_person_file
        test_folder = os.path.join(student_folder, f'test_{self.current_test_number}')
        speed_folder = os.path.join(test_folder, 'speed')
        
        # 确保文件夹存在
        os.makedirs(speed_folder, exist_ok=True)
        
        # 弹出文件选择器选择视频文件
        video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.avi *.mp4)")
        
        if video_path:
            # 重命名视频文件为speed_file_name
            speed_file_name = os.path.join(speed_folder, "SpeedVideo.avi")
            shutil.copy(video_path, speed_file_name)
            self.show_video_preview_image()

    def load_jump_video(self):
        """加载跳远视频"""
        if not hasattr(self, 'sele_person_file') or not self.sele_person_file:
            QMessageBox.warning(self, "提示", "请先选择学生")
            return
        
        # 使用当前测试次数获取文件夹路径
        student_folder = self.sele_person_file
        test_folder = os.path.join(student_folder, f'test_{self.current_test_number}')
        jump_folder = os.path.join(test_folder, 'jump')
        
        # 确保文件夹存在
        os.makedirs(jump_folder, exist_ok=True)
        
        # 弹出文件选择器选择视频文件
        video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.avi *.mp4)")
        
        if video_path:
            # 重命名视频文件为jump_file_name
            jump_file_name = os.path.join(jump_folder, "JumpVideo.avi")
            shutil.copy(video_path, jump_file_name)
            self.show_video_preview_image()

    def load_stuscore(self):
        self.stu=StudentInfoTable(self.sele_person_file)
        self.stu.show()

    # def initposture(self):
    #     self.initposture_window = initposture_MainWindow()
    #     self.initposture_window.resize(1600, 900)

    #     self.initposture_window.show()




    def on_student_clicked(self, item):
        """处理学生选择"""
        # 停止之前的显示线程
        if self.pose_display_thread and self.pose_display_thread.isRunning():
            self.pose_display_thread.stop()
            self.pose_display_thread.wait()
            self.Ca_main.clear()  # 清除显示
            
        row = item.row()
        table = item.tableWidget()
        
        # 获取学生信息
        student_id = table.item(row, 0).text()
        name = table.item(row, 1).text()
        
        # 更新界面显示
        self.name_label.setText(name)
        self.sno_label.setText(student_id)
        
        # 建学生文件夹路径（不创建新的评文件夹）
        speed_folder, jump_folder = self.student_manager.create_student_folders(create_new=False)
        # 设置当前选中学生的文件夹路径
        self.sele_person_file = os.path.dirname(os.path.dirname(speed_folder))
        
        # 视频预览
        self.show_video_preview_image()
        
        # 加载学生分数
        self.student_manager.load_student_scores()

        # 加载已有的得分数据
        self.load_existing_scores()

   

   

    def create_folder(self):
        """ 创建结果文件夹，如果不存在则创建 """
        self.results_folder = './results'
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
        


    def toggle_cameras(self):
        if self.camera1.running or self.camera2.running:
            self.camera1.stop_camera()
            self.camera2.stop_camera()
            self.timer.stop()
            self.Preview.setText("预览")
        else:
            self.camera1.start_camera()
            self.current_display = "camera1"
            self.timer.start(30)
            self.Preview.setText("停止")

    def show_camera1_view(self):
        if self.camera2.running:
            self.camera2.stop_camera()
        if not self.camera1.running:
            self.camera1.start_camera()
        self.current_display = "camera1"

    def show_camera2_view(self):
        if self.camera1.running:
            self.camera1.stop_camera()
        if not self.camera2.running:
            self.camera2.start_camera()
        self.current_display = "camera2"

    def update_images(self):
            if self.current_display == "camera1":
                frame = self.camera1.get_frame()
            elif self.current_display == "camera2":
                frame = self.camera2.get_frame()

            if frame is not None:
                image =convert_cv_qt(frame, 640,360)
                self.Ca_main.setPixmap(QPixmap.fromImage(image))

    def star_Recording(self):
        """开始测评"""
        try:
            # 如果当前是录制状态，执行停止录制
            if self.Star.text() == "停止录制":
                try:
                    student_id = self.sno_label.text()  # 获取当前学生ID
                    
                    # 停止录制
                    if self.recorder1.running:
                        print(f"正在保存速度视频到: {self.speed_file}")
                        self.recorder1.stop_recording(self.speed_camera_id)
                        if os.path.exists(self.speed_file):
                            print(f"速度视频已保存: {self.speed_file}")
                        else:
                            print(f"警告: 速度视频文件未找到: {self.speed_file}")
                        self.student_manager.update_student_status(student_id, "处理中")
                    
                    if self.recorder2.running:
                        print(f"正在保存跳远视频到: {self.jump_file}")
                        self.recorder2.stop_recording(self.jump_camera_id)
                        if os.path.exists(self.jump_file):
                            print(f"跳远视频已保存: {self.jump_file}")
                        else:
                            print(f"警告: 跳远视频文件未找到: {self.jump_file}")
                        self.student_manager.update_student_status(student_id, "处理中")
                    
                    # 停止计时器
                    self.timer.stop()
                    self.update_timer.stop()
                    self.timer_started = False
                    self.elapsed_time = 0
                    
                    # 恢复按钮样式和文本为蓝色开始状态
                    self.Star.setStyleSheet("""
                        QPushButton {
                            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                      stop:0 #4B6EAF, stop:1 #182B4D);
                            border: 1px solid #1C3053;
                            border-radius: 5px;
                            color: white;
                            padding: 5px 15px;
                        }
                    """)
                    self.Star.setText("开始录制")

                    # 更新视频预览
                    self.show_video_preview_image()
                    
                    return
                except Exception as e:
                    print(f"停止录制时出错: {str(e)}")
                    traceback.print_exc()
                    return

            # 开始录制的逻辑
            # 检查是否选择了学生
            if not hasattr(self, 'sele_person_file'):
                QMessageBox.warning(self, "提示", "请先选择学生")
                return

            # 获取当前选中的学生ID
            student_id = self.sno_label.text()
            if not student_id:
                QMessageBox.warning(self, "提示", "未获取到学生ID")
                return

            # 创建新的测评文件夹
            speed_folder, jump_folder = self.student_manager.create_student_folders(create_new=True)
            if not speed_folder or not jump_folder:
                return
            
            # 简化视频文件命名，因为已经在对应测试次数的文件夹中
            speed_file_name = os.path.join(speed_folder, "SpeedVideo.avi")
            jump_file_name = os.path.join(jump_folder, "JumpVideo.avi")
            
            # 如果当前是停止状态，开始录制
            if self.Star.text() == "开始录制":
                # 检查摄像头是否可用
                camera1_available = self.camera1.check_camera_available()
                camera2_available = self.camera2.check_camera_available()
                
                if not camera1_available and not camera2_available:
                    QMessageBox.warning(self, "错误", "未检测到可用的摄像头")
                    return
                
                # 改变按钮式和文本为红色停止状态
                self.Star.setStyleSheet("""
                    QPushButton {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                  stop:0 #D93240, stop:1 #AB1E2A);
                        border: 1px solid #8B1823;
                        border-radius: 5px;
                        color: white;
                        padding: 5px 15px;
                    }
                    QPushButton:hover {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                  stop:0 #E84250, stop:1 #BC2F3A);
                    }
                    QPushButton:pressed {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                  stop:0 #AB1E2A, stop:1 #D93240);
                    }
                """)
                self.Star.setText("停止录制")
                
                # 开始录制
                if not self.recorder1.running and not self.recorder2.running:
                    self.speed_file = speed_file_name
                    self.jump_file = jump_file_name
                    
                    # 启动录像机
                    recording_started = False
                    
                    if self.recorder1_open and camera1_available:
                        try:
                            # 确保目录存在
                            os.makedirs(speed_folder, exist_ok=True)
                            self.recorder1.start_recording(self.speed_camera_id, speed_file_name)
                            recording_started = True
                        except Exception as e:
                            print(f"启动速度摄像头失败: {str(e)}")
                    
                    if self.recorder2_open and camera2_available:
                        try:
                            # 确保目录存在
                            os.makedirs(jump_folder, exist_ok=True)
                            self.recorder2.start_recording(self.jump_camera_id, jump_file_name)
                            recording_started = True
                        except Exception as e:
                            print(f"启动跳远摄像头失败: {str(e)}")
                    
                    if recording_started:
                        # 启动计时器
                        self.timer_started = True
                        self.elapsed_time = 0
                        self.update_timer.start(1000)
                        self.timer.start(30)
                    else:
                        QMessageBox.warning(self, "错误", "无法启动录制")
                        self.Star.setText("开始录制")
                        # 恢复按钮样式
                        self.Star.setStyleSheet("""
                            QPushButton {
                                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                          stop:0 #4B6EAF, stop:1 #182B4D);
                                border: 1px solid #1C3053;
                                border-radius: 5px;
                                color: white;
                                padding: 5px 15px;
                            }
                        """)
                    
            # 如果当前是录制状态，停止录制
            else:
                try:
                    # 停止录制
                    if self.recorder1.running:
                        print(f"正在保存速度视频到: {self.speed_file}")  # 添加日志
                        self.recorder1.stop_recording(self.speed_camera_id)
                        if os.path.exists(self.speed_file):
                            print(f"速度视频已保存: {self.speed_file}")
                        else:
                            print(f"警告: 速度视频文件未找到: {self.speed_file}")
                        self.student_manager.update_student_status(self.speed_file,"处理中")  # 更新学生状态
                    
                    if self.recorder2.running:
                        print(f"正在保存跳远视频到: {self.jump_file}")  # 添加日志
                        self.recorder2.stop_recording(self.jump_camera_id)
                        if os.path.exists(self.jump_file):
                            print(f"跳远视频已保存: {self.jump_file}")
                        else:
                            print(f"警告: 跳远视频文件未找到: {self.jump_file}")
                        self.student_manager.update_student_status(self.jump_file,"处理中")  # 更新学生状态
                    
                    # 停止计时器
                    self.timer.stop()
                    self.update_timer.stop()
                    self.timer_started = False
                    self.elapsed_time = 0
                    
                    # 恢复按钮样式和文本为蓝色开始状态
                    self.Star.setStyleSheet("""
                        QPushButton {
                            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                      stop:0 #4B6EAF, stop:1 #182B4D);
                            border: 1px solid #1C3053;
                            border-radius: 5px;
                            color: white;
                            padding: 5px 15px;
                        }
                    """)
                    self.Star.setText("开始录制")
                except Exception as e:
                    print(f"停止录制时出错: {str(e)}")
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"录制过程出错: {str(e)}")
            traceback.print_exc()
            QMessageBox.warning(self, "错误", f"录制过程出错: {str(e)}")

    def stop_Recording(self):  # 停止测评，录制视频并保存


        if self.timer_started:
            # 恢复按钮样式和文本
            self.Star.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                              stop:0 #4B6EAF, stop:1 #182B4D);
                    border: 1px solid #1C3053;
                    border-radius: 5px;
                    color: white;
                    padding: 5px 15px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                              stop:0 #5680C9, stop:1 #1F3866);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                              stop:0 #182B4D, stop:1 #4B6EAF);
                }
            """)
            self.Star.setText("开始录制")
            
            self.update_timer.stop()
            self.timer_started = False
            self.elapsed_time = 0
            

        recorder1_stopped = False
        recorder2_stopped = False

        # 单独查每个像是否在运行，如果是，停止它
        if self.recorder1.running:
            self.recorder1.stop_recording(self.speed_camera_id)
            recorder1_stopped = True  # 记recorder1已停止
            self.student_manager.update_student_status(self.speed_file,"处理中")  # 更新学生状态
           

        if self.recorder2.running:
            self.recorder2.stop_recording(self.jump_camera_id)
            recorder2_stopped = True  # 标记recorder2已停止
            self.student_manager.update_student_status(self.jump_file,"处理中")  # 更新学生状态
            

        # 如果任何一个录像机被停止，进行以下操作
        if recorder1_stopped or recorder2_stopped:
            self.timer.stop()  # 停止计时器
            self.Star.setEnabled(True)
            self.Stop.setEnabled(False)

   

    def Star_test(self):
        """开始测评"""
        speed_file_name, jump_file_name = get_video_file_paths(self.student_manager)
        student_id = self.sno_label.text()

        if not student_id:
            print("未选择学生")
            return

        # 更改按钮状态和文本
        self.Star_test_but.setText("测评中...")
        self.Star_test_but.setEnabled(False)  # 禁用按钮

        # 首先更新学生状态为处理中
        self.student_manager.update_student_status(student_id, "测评中", 0)

        # 开始显示实时处理画面
        if os.path.exists(jump_file_name):
            # 停止之前的显示线程（如果有）
            if self.pose_display_thread and self.pose_display_thread.isRunning():
                self.pose_display_thread.stop()
                self.pose_display_thread.wait()

            # 创建新的显示线程
            self.current_display_student_id = student_id
            self.pose_display_thread = PoseDisplayThread(jump_file_name, student_id, save_video=True)
            self.pose_display_thread.frame_ready.connect(self.update_display_frame)
            self.pose_display_thread.start()

            # 处理跳远视频
            try:
                self.is_jump_video_processing = True
                
                # 创建并启动姿态分析线程
                self.pose_thread = PoseAnalysisThread(
                    self.jump_analyzer,
                    self.pose_matcher,
                    jump_file_name,
                    "public/data/standard_poses/stageroot.json",
                    student_id
                )
                
                # 连接信号
                self.pose_thread.progress.connect(lambda msg: print(msg))
                self.pose_thread.progress_value.connect(self.on_progress_update)
                self.pose_thread.finished.connect(lambda: (
                    self.on_pose_analysis_finished  # 调用原有的完成回调
                ))
                
                # 启动线程
                self.pose_thread.start()
                
            except Exception as e:
                print(f"处理跳远视频时出错: {str(e)}")
                self.is_jump_video_processing = False
                self.student_manager.update_student_status(student_id, "未测评")

        # 处理速度视频
        if not self.is_speed_video_processing and os.path.exists(speed_file_name):
            try:
                self.is_speed_video_processing = True
                print(f"开始处理速度视频: {speed_file_name}")
                self.speed = Speed_VideoProcessor(speed_file_name)
                self.speed.processingFinished.connect(self.on_speed_video_processed)
                self.speed.start()
            except Exception as e:
                print(f"处理速度视频时出错: {str(e)}")
                self.is_speed_video_processing = False

    def update_display_frame(self, pixmap):
        """更新显示帧"""
        if self.current_display_student_id == self.sno_label.text():
            scaled_pixmap = pixmap.scaled(
                self.Ca_main.width(),
                self.Ca_main.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.Ca_main.setPixmap(scaled_pixmap)

    def on_progress_update(self, progress_value, student_id):
        """处理进度更新"""
        # 只在进度值变化较大时打比如每10%打印次
        if progress_value % 10 == 0:
            print(f"处理进度: {progress_value}%")
        self.student_manager.update_progress(progress_value, student_id)

    def on_pose_analysis_finished(self, result):
        """姿态分析完成的回调"""
        try:
            success, test_number = result  # 解包元组
            self.is_jump_video_processing = False
            student_id = self.sno_label.text()
            
            if success:
                print("\n=== 开始更新学生信息 ===")
                # 使用绝对路径
                excel_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'student', 'student.xlsx'))
                print(f"Excel文件路径: {excel_file}")
                
                if os.path.exists(excel_file):
                    print("找到Excel文件")
                    # 读取Excel文件
                    df = pd.read_excel(excel_file)
                    print(f"Excel列名: {df.columns.tolist()}")
                    print(f"要更新的学生ID: {student_id}")
                    
                    # 确保student_id是字符串类型
                    df['student_id'] = df['student_id'].astype(str)
                    student_id = str(student_id)
                    
                    print(f"Excel中的学生ID列: {df['student_id'].tolist()}")
                    
                    mask = df['student_id'] == student_id
                    if any(mask):
                        print("找到匹配的学生记录")
                        print(f"更新前的记录: \n{df[mask]}")
                        
                        # 更新数据
                        df.loc[mask, 'test_count'] = int(test_number)
                        df.loc[mask, 'status'] = '已测评'
                        
                        print(f"更新后的记录: \n{df[mask]}")
                        
                        try:
                            # 保存更新后的Excel文件
                            df.to_excel(excel_file, index=False)
                            print("已保存更新后的Excel文件")
                            
                            # 直接更新界面显示，而不是重导入
                            self.student_manager.update_student_display()  # 新方法
                            print("学生列表已刷新")
                        except PermissionError:
                            print("错误: 无法保存Excel文件，可能是文件被其他程序占用")
                        except Exception as e:
                            print(f"保存Excel文件时出错: {str(e)}")
                            traceback.print_exc()
                    else:
                        print(f"警告: 在Excel中未找到学生: {student_id}")
                else:
                    print(f"错误: Excel文件不存在: {excel_file}")
                
                print("=== 学生信息更新完成 ===")
                
                # 更新学生状态
                self.student_manager.update_student_status(student_id, "已测评")
            else:
                print("姿态分析失败")
                self.student_manager.update_student_status(student_id, "未测评")
                
                # 分析失败时重置得分显示
                self.labelTakeOffScore.setText("0.00")
                self.labelHipExtensionScore.setText("0.00")
                self.labelAbdominalContractionScore.setText("0.00")
                self.labelAllScore.setText("0.00")
                
        except Exception as e:
            print(f"更新学生信息时出错: {str(e)}")
            traceback.print_exc()

    def on_speed_video_processed(self, processed_video_path):
        try:
            print(f"速度处理完成: {processed_video_path}")
            self.is_speed_video_processing = False
            
            # 获取速度结果文件路径
            speed_result_file = os.path.join(os.path.dirname(processed_video_path), 'speed_result.json')
            
            if os.path.exists(speed_result_file):
                with open(speed_result_file, 'r', encoding='utf-8') as f:
                    speed_data = json.load(f)
                    mean_speed = speed_data.get('mean_speed', 0)
                    # 更新速度显示
                    self.labelMeanVector.setText(f"{mean_speed:.2f} m/s")
            
            # 更新学生状态
            student_id = self.sno_label.text()
            self.student_manager.update_student_status(student_id, "已处理")
        except Exception as e:
            print(f"更新速度显示时出错: {str(e)}")
            traceback.print_exc()

    def on_jump_video_processed(self, processed_video_path):
        """跳远视频处理完成的回调"""
        print(f"跳远处理完成: {processed_video_path}")
        self.is_jump_video_processing = False
        student_id = self.sno_label.text()
        self.student_manager.update_student_status(student_id, "已处理")

    def onProcessingFinished(self):
        self.jump.quit() 
        self.jump.wait()


    def setBoard_camera_source(self):
        if self.camera1.running or  self.camera2.running:
            self.camera1.stop_camera()
            self.camera2.stop_camera() 


        frame1 = self.camera1.get_frame()
        # 建图像标记窗口
        self.image_window = ImageWindow(frame1)
        self.image_window.show()

        pixmap = QPixmap("calibrated_image.jpg")
        self.Ca_calibration.setPixmap(pixmap)
        self.Ca_calibration.setScaledContents(True)

            
    def setBoard_video_source(self):
      
          
        # 弹出文件选择器选择视频文件
        video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.avi *.mp4)")

        cap = cv2.VideoCapture(video_path)
        success, frame1 = cap.read()

        # 创建图像标记窗口
        self.image_window = ImageWindow(frame1)
        self.image_window.show()

        cap.release()

        pixmap = QPixmap("calibrated_image.jpg")
        self.Ca_calibration.setPixmap(pixmap)
        self.Ca_calibration.setScaledContents(True)

    def results(self):
        """显示结果窗口"""
        if hasattr(self, 'sele_person_file') and self.sele_person_file:
            self.show_results = ShowResults(self.sele_person_file)
            # 确保当前测试次数被正确传递
            self.show_results.current_test_number = self.current_test_number
            self.show_results.show()
        else:
            QMessageBox.warning(self, "提示", "请先选择学生")

    def closeEvent(self, event):
        """窗口关闭时的处理"""
        # 停止显示线程
        if self.pose_display_thread and self.pose_display_thread.isRunning():
            self.pose_display_thread.stop()
            self.pose_display_thread.wait()
            
        self.camera1.stop_camera()
        self.camera2.stop_camera()
        self.timer.stop()
        super().closeEvent(event)

    def update_timer_label(self):
        """更新计时器标签显示"""
        try:
            self.elapsed_time += 1
            elapsed_time_qt = QTime(0, 0, 0).addSecs(self.elapsed_time)
            self.timerLabel.setText(elapsed_time_qt.toString("hh:mm:ss"))
        except Exception as e:
            print(f"更新计时器显示时出错: {str(e)}")
            traceback.print_exc()

    def setup_student_manager(self):
        """设置学生管理"""
        self.student_manager = StudentManager(self)
        
        # 连接格点击信号
        for table in [self.studentList_1, self.studentList_2, self.studentList_3]:
            table.itemClicked.connect(self.on_student_clicked)

    def switch_test_number(self, test_number):
        """切换测试次数"""
        # 停止之前的显示线程
        if self.pose_display_thread and self.pose_display_thread.isRunning():
            self.pose_display_thread.stop()
            self.pose_display_thread.wait()
            self.Ca_main.clear()  # 清除显示
            
        if not hasattr(self, 'sele_person_file') or not self.sele_person_file:
            QMessageBox.warning(self, "提示", "请先选择学生")
            return
        
        # 更新当前测试次数
        self.current_test_number = test_number
        
        student_folder = self.sele_person_file
        test_folder = os.path.join(student_folder, f'test_{test_number}')
        
        # 创建测文件夹（如果不存在）
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
            os.makedirs(os.path.join(test_folder, 'speed'))
            os.makedirs(os.path.join(test_folder, 'jump'))
        
        # 更新视频预览
        self.show_video_preview_image()
        
        # 更新成绩显示
        self.student_manager.load_student_scores()
        
        # 高亮当前选中的按钮
        self.but_1.setStyleSheet("" if test_number != 1 else "background-color: lightblue;")
        self.but_2.setStyleSheet("" if test_number != 2 else "background-color: lightblue;")
        self.but_3.setStyleSheet("" if test_number != 3 else "background-color: lightblue;")

        # 加载对应测试次数的得分
        self.load_existing_scores()

    def update_student_info(self, test_number):
        """更新学生信息"""
        try:
            print("\n=== 开始更新学生信息 ===")
            if hasattr(self, 'sele_person_file'):
                # 获取学生ID（从文件夹称）
                student_id = os.path.basename(self.sele_person_file)
                print(f"学生ID: {student_id}")
                print(f"测试次数: {test_number}")
                
                # 获取学生信息文件路径
                student_info_file = os.path.join(self.sele_person_file, 'student_info.json')
                print(f"学生信息文件路径: {student_info_file}")
                
                if os.path.exists(student_info_file):
                    print("找到学生信息文件")
                    # 读取当前信息
                    with open(student_info_file, 'r', encoding='utf-8') as f:
                        student_info = json.load(f)
                    print(f"当前学生信息: {student_info}")
                    
                    # 确保student_info包含id
                    student_info['student_id'] = student_id
                    student_info['test_count'] = int(test_number)
                    student_info['status'] = '已测评'
                    student_info['last_test_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    print(f"更新后的学生信息: {student_info}")
                    
                    # 保存更新后的信息
                    with open(student_info_file, 'w', encoding='utf-8') as f:
                        json.dump(student_info, f, indent=4, ensure_ascii=False)
                    print("已保存更新后的学生信息文件")
                    
                    # 更新Excel文件
                    self.update_student_excel(student_info)
                else:
                    print(f"警告: 未找到学生信息文件: {student_info_file}")
            else:
                print("错误: sele_person_file 未设置")
            
            print("=== 学生信息更新完成 ===\n")
        
        except Exception as e:
            print("\n=== 更新学生信息时出错 ===")
            print(f"错误信息: {str(e)}")
            import traceback
            traceback.print_exc()
            print("=== 错误信息结束 ===\n")

    def update_student_excel(self, student_info):
        """更新学生Excel文件"""
        try:
            print("\n=== 开始更新Excel文件 ===")
            excel_file = os.path.join('student', 'student.xlsx')  # 使用正确的路径
            print(f"Excel文件路径: {excel_file}")
            
            if os.path.exists(excel_file):
                print("找到Excel文件")
                # 读取Excel文件
                df = pd.read_excel(excel_file)
                print(f"Excel列: {df.columns.tolist()}")
                
                # 查找并更新学生信息
                student_id = student_info.get('student_id')
                print(f"要更新的学生ID: {student_id}")
                
                if student_id:
                    # 确保student_id是字符串类型
                    df['student_id'] = df['student_id'].astype(str)
                    student_id = str(student_id)
                    
                    print(f"Excel中的学生ID列: {df['student_id'].tolist()}")
                    
                    mask = df['student_id'] == student_id
                    if any(mask):
                        print("找到匹配的学生记录")
                        print(f"更新前的记录: \n{df[mask]}")
                        
                        # 更新数据
                        df.loc[mask, 'test_count'] = int(student_info.get('test_count', 0))
                        df.loc[mask, 'status'] = student_info.get('status', '已测评')
                        
                        print(f"更新后的记录: \n{df[mask]}")
                        
                        try:
                            # 尝试保存文件
                            df.to_excel(excel_file, index=False)
                            print("已保存更新后的Excel文件")
                        except PermissionError:
                            print("错误: 无法保存Excel文件，可能是文件被其他程序占用")
                        except Exception as e:
                            print(f"保存Excel文件时出错: {str(e)}")
                        
                        # 制刷新显示
                        print("正在刷新学生列表显示...")
                        self.student_manager.import_students()
                        print("学生列表已刷新")
                    else:
                        print(f"警告: 在Excel中未找到学生: {student_id}")
                        print("可用的学生ID:")
                        print(df['student_id'].tolist())
                else:
                    print("错误: student_info 中没有 student_id")
            else:
                print(f"错误: Excel文件不存在: {excel_file}")
            
            print("=== Excel文件更新完成 ===\n")
        
        except Exception as e:
            print("\n=== 更新Excel文件时出错 ===")
            print(f"错误信息: {str(e)}")
            import traceback
            traceback.print_exc()
            print("=== 错误信息结束 ===\n")

    def load_existing_scores(self):
        """加载已存在的得分数据"""
        try:
            if hasattr(self, 'sele_person_file') and self.current_test_number:
                # 加载速度得分
                mean_vector_path = os.path.join(self.sele_person_file, f'test_{self.current_test_number}', 'speed', 'mean_vector.txt')
                if os.path.exists(mean_vector_path):
                    with open(mean_vector_path, 'r', encoding='utf-8') as file:
                        mean_speed = file.read().strip()  # 读取并去除空白字符
                        self.labelMeanVector.setText(f"{mean_speed} m/s")  # 显示速度
                        print(f"已加载速度: {mean_speed} m/s")  # 调试输出
                else:
                    print(f"速度文件不存在: {mean_vector_path}")
                
                # 加载跳远得分
                jump_score_file = os.path.join(
                    self.sele_person_file, 
                    f'test_{self.current_test_number}', 
                    'jump', 
                    'jump_score_result.json'
                )
                if os.path.exists(jump_score_file):
                    with open(jump_score_file, 'r', encoding='utf-8') as f:
                        score_data = json.load(f)
                        
                        # 从phase_scores中获各阶段得分
                        phase_scores = score_data.get('phase_scores', {})
                        
                        # 获取各阶段得分
                        takeoff_score = phase_scores.get('takeoff', {}).get('score', 0)
                        flight_score = phase_scores.get('flight', {}).get('score', 0)
                        landing_score = phase_scores.get('landing', {}).get('score', 0)
                        overall_score = score_data.get('overall_score', 0)
                        
                        # 更新界面显示
                        self.labelTakeOffScore.setText(f"{takeoff_score:.2f}")
                        self.labelHipExtensionScore.setText(f"{flight_score:.2f}")
                        self.labelAbdominalContractionScore.setText(f"{landing_score:.2f}")
                        self.labelAllScore.setText(f"{overall_score:.2f}")
                else:
                    # 如果文件不存在，重置所有得分显示
                    self.labelTakeOffScore.setText("0.00")
                    self.labelHipExtensionScore.setText("0.00")
                    self.labelAbdominalContractionScore.setText("0.00")
                    self.labelAllScore.setText("0.00")
                
        except Exception as e:
            print(f"加载已有得分时出错: {str(e)}")
            traceback.print_exc()


# 添加新的线程类
class PoseAnalysisThread(QThread):
    finished = Signal(tuple)  # 修改发送元组 (success: bool, test_number: int)
    progress = Signal(str)
    progress_value = Signal(int, str)
    
    def __init__(self, jump_analyzer, pose_matcher, video_path, stage_file_path, student_id):
        super().__init__()
        self.jump_analyzer = jump_analyzer
        self.pose_matcher = pose_matcher
        self.video_path = video_path
        self.stage_file_path = stage_file_path
        self.save_folder = os.path.dirname(video_path)
        self.student_id = student_id
        self.total_steps = 4  # 总步骤数（视频处理 + 3个姿态匹配）
        self.current_step = 0

    def update_progress(self, progress):
        """新进度"""
        # 在关键阶段打进度
        if progress in [0, 25, 50, 75, 100]:
            print(f"\n当前阶段: {self.get_stage_description(progress)}")
        self.progress_value.emit(progress, self.student_id)
    
    def get_stage_description(self, progress):
        """获取当前段描述"""
        if progress == 0:
            return "开始处理"
        elif progress <= 25:
            return "视频处理"
        elif progress <= 50:
            return "起跳姿态分析"
        elif progress <= 75:
            return "空中姿态分析"
        else:
            return "落地姿态分析"

    def run(self):
        try:
            self.progress.emit("开始处理跳远视频...")
            
            # 视频处理阶段
            self.update_progress(0)
            self.jump_analyzer.set_start_zone(100, 100, 300, 200)
            
            # 处理视频帧获取进度
            frame_infos = []  # 存储所有帧的信息
            for progress, frame, frame_info in self.jump_analyzer.process_frame(self.video_path, 0):
                if frame_info:  # 确保frame_info不为空
                    frame_info['frame_idx'] = frame_info['frame_id']  # 确保有frame_idx字段
                    frame_infos.append(frame_info)
                self.update_progress(progress)
            
            # 修改保存的数据格式
            output_json_path = os.path.join(self.save_folder, "jump_analysis.json")
            analysis_data = {
                'version': '1.0',
                'data': {
                    'frames': [
                        {
                            'frame_idx': info['frame_id'],
                            'keypoints': info['keypoints'],
                            'scores': info['scores'],
                            'bbox': info['bbox']
                        }
                        for info in frame_infos
                    ],
                    'total_frames': len(frame_infos)
                }
            }
            
            # 保存为JSON文件
            with open(output_json_path, 'w') as f:
                json.dump(analysis_data, f, indent=4)
            print(f"\n分析结果已保存到: {output_json_path}")
            
            self.update_progress(30)
            
            try:
                # 加载阶段划分文件
                with open(self.stage_file_path, 'r', encoding='utf-8') as f:
                    stage_data = json.load(f)
                self.update_progress(35)
                
                # 从阶段数据中获取每个阶段的信息
                stage_info = {}
                for stage in stage_data['stages']:
                    stage_info[stage['id']] = {
                        'name': stage['name'],
                        'range': [
                            int(stage['info']['pixel_start']),
                            int(stage['info']['pixel_end'])
                        ]
                    }
                    print(f"找到阶段: {stage['id']}, 范围: {stage_info[stage['id']]['range']}")
                
                # 定义要检测的姿态配置
                pose_configs = [
                    {
                        'stage_id': 'takeoff',
                        'folder_name': 'takeoff',
                        'standard_pose_path': 'public/data/standard_poses/takeoff/root.json',
                        'description': '起跳姿态',
                        'range': stage_info['takeoff']['range'] if 'takeoff' in stage_info else None
                    },
                    {
                        'stage_id': 'flight',
                        'folder_name': 'flight',
                        'standard_pose_path': 'public/data/standard_poses/flight/root.json',
                        'description': '空中姿态',
                        'range': stage_info['flight']['range'] if 'flight' in stage_info else None
                    },
                    {
                        'stage_id': 'landing',
                        'folder_name': 'landing',
                        'standard_pose_path': 'public/data/standard_poses/landing/root.json',
                        'description': '落地姿态',
                        'range': stage_info['landing']['range'] if 'landing' in stage_info else None
                    }
                ]

                # 对每个姿态进行匹配
                all_scores = {}
                for i, pose_config in enumerate(pose_configs):
                    print(f"\n开始处理 {pose_config['description']} 阶段")
                    print(f"阶段范围: {pose_config['range']}")
                    
                    if pose_config['range'] is None:
                        print(f"警告: {pose_config['description']} 阶段范围未定义，跳过处理")
                        continue
                    
                    if not os.path.exists(pose_config['standard_pose_path']):
                        print(f"错误: 标准姿态文件不存在: {pose_config['standard_pose_path']}")
                        continue

                    self.progress.emit(f"开始匹配{pose_config['description']}...")
                    base_progress = 35 + (i * 20)
                    
                    try:
                        matches = self.pose_matcher.find_best_matches_in_stage(
                            pose_config['standard_pose_path'],
                            output_json_path,
                            self.stage_file_path,
                            pose_config['stage_id']
                        )
                        
                        if matches:
                            print(f"{pose_config['description']} 匹配成功，找到 {len(matches)} 个匹配")
                            # 建姿态特的文件夹
                            pose_dir = os.path.join(self.save_folder, pose_config['folder_name'])
                            os.makedirs(pose_dir, exist_ok=True)
                            
                            # 保存匹配结果
                            match_result_path = os.path.join(pose_dir, 'pose_matches.json')
                            match_data = {
                                'version': '1.0',
                                'stage_id': pose_config['stage_id'],
                                'matches': matches
                            }
                            with open(match_result_path, 'w') as f:
                                json.dump(match_data, f, indent=4)
                            
                            self.update_progress(base_progress + 15)
                            
                            # 保存匹配帧到对姿态文件夹
                            self.pose_matcher.save_matched_frames(
                                self.video_path, 
                                matches, 
                                pose_dir,
                                'compare_'  # 添加前缀
                            )
                            
                            # 修改这里：从列表获取最佳匹配的分数
                            best_match = matches[0] if matches else {'similarity': 0}
                            all_scores[pose_config['stage_id']] = best_match['similarity']
                            
                            self.update_progress(base_progress + 20)
                            
                    except Exception as e:
                        print(f"匹配 {pose_config['description']} 时出错: {str(e)}")
                        traceback.print_exc()
                        continue
                
                # 保存总分数
                if all_scores:
                    score_file = os.path.join(self.save_folder, 'score_jump.json')
                    final_scores = {
                        'take_off_score': all_scores.get('takeoff', 0),
                        'hip_extension_score': all_scores.get('flight', 0),
                        'abdominal_contraction_score': all_scores.get('landing', 0),
                        'all_score': sum(all_scores.values()) / len(all_scores)
                    }
                    with open(score_file, 'w') as f:
                        json.dump(final_scores, f, indent=4)
                
                self.update_progress(100)
                
                # 在姿态匹配完成后，加评分计算
                try:
                    # 创建分计算器实例
                    calculator = KeyScoreCalculator()
                    
                    # 计算各阶段分数
                    scores = calculator.calculate_all_scores(self.save_folder)
                    print("\n各段得分:")
                    for pose_type, score in scores.items():
                        print(f"{pose_type}: {score:.2f}")
                    
                    # 生成最跳远评分文件
                    calculator.generate_jump_score(self.save_folder)
                    
                    print("评分计算完成")
                    
                except Exception as e:
                    print(f"评分计算时出错: {str(e)}")
                    traceback.print_exc()
                
                # 获取当前测试次数
                test_folder = os.path.dirname(self.save_folder)
                test_number = int(os.path.basename(test_folder).split('_')[1])
                
                # 发送信号以更新学生信息
                self.finished.emit((True, test_number))
                
            except Exception as e:
                print(f"加载阶段数据时出错: {str(e)}")
                traceback.print_exc()
                self.finished.emit((False, 0))  # 失败时发送0作为测试次数
                
        except Exception as e:
            self.progress.emit(f"处理出错: {str(e)}")
            traceback.print_exc()
            self.finished.emit((False, 0))  # 失败时发送0作为��试次数


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
