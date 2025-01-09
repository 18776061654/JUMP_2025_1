import os
import json
import cv2
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QPushButton
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QThread, Signal, QFile, Qt, QObject
from PySide6.QtUiTools import QUiLoader
from functools import partial
from main_part.pose_match_details import PoseMatchDetails
from main_part.video_frame_selector import VideoFrameSelector
import pandas as pd

class VideoThread(QThread):
    change_pixmap_signal = Signal(QPixmap)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.running = True
        self.paused = False
        
        # 获取视频的原始帧率
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # 计算每帧之间需要等待的时间（毫秒）
        self.frame_delay = int(1000.0 / self.fps)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while self.running and cap.isOpened():
            if self.paused:
                self.msleep(100)
                continue
                
            ret, frame = cap.read()
            if ret:
                # 转换并发送帧
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = QPixmap.fromImage(convert_to_Qt_format).scaled(521, 361, Qt.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)
                
                # 按原始帧率延时
                self.msleep(self.frame_delay)
            else:
                # 视频播放完毕，重新开始
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
        cap.release()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.running = False
        self.paused = False

class ShowResults(QMainWindow):
    def __init__(self,sele_person_file):
        super(ShowResults, self).__init__()
        ui_file = QFile('public/resources/newwindow/result.ui')
        ui_file.open(QFile.ReadOnly)
        self.ui = QUiLoader().load(ui_file, self)
        ui_file.close()
        self.resize(1319, 736)

        # 打印所有按钮名称（调试用）
        print("\n可用的按钮名称:")
        for button in self.ui.findChildren(QPushButton):
            if button.objectName():
                print(f"- {button.objectName()}")

        self.base_path = './results'
        self.sele_person_file = sele_person_file
        
        # 从主窗口获取当前测试次数
        self.current_test_number = QApplication.activeWindow().current_test_number
        
        self.current_mode = 'jump'
        
        # 连接按钮点击事件
        self.setup_button_connections()
        
        self.load_media()

        # self.ui.PreBtn.clicked.connect(self.prev_media)
        # self.ui.NextBtn.clicked.connect(self.next_media)
        self.ui.SwitchBasePathBtn.clicked.connect(self.switch_video_mode)
        self.ui.StartBtn.clicked.connect(self.toggle_pause_video)

        self.ui.Selet_takeoffBtn.clicked.connect(self.Selet_takeoff)
        self.ui.Selet_hipBtn.clicked.connect(self.Selet_hip)
        self.ui.Selet_abdBtn.clicked.connect(self.Selet_abd)

        # 添加一个属性来保持对详细窗口的引用
        self.pose_details_windows = {}

    def Selet_takeoff(self):
        """选择起跳阶段帧"""
        try:
            test_folder = os.path.join(self.sele_person_file, f'test_{self.current_test_number}', 'jump', 'takeoff')
            self.selector = VideoFrameSelector('takeoff', test_folder)
            self.selector.imageUpdated.connect(self.load_media)
            self.selector.show()
        except Exception as e:
            print(f"选择起跳帧时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"选择起跳帧时出错: {str(e)}")

    def Selet_hip(self):
        """选择空中阶段帧"""
        try:
            test_folder = os.path.join(self.sele_person_file, f'test_{self.current_test_number}', 'jump', 'flight')
            self.selector = VideoFrameSelector('flight', test_folder)
            self.selector.imageUpdated.connect(self.load_media)
            self.selector.show()
        except Exception as e:
            print(f"选择空中帧时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"选择空中帧时出错: {str(e)}")

    def Selet_abd(self):
        """选择落地阶段帧"""
        try:
            test_folder = os.path.join(self.sele_person_file, f'test_{self.current_test_number}', 'jump', 'landing')
            self.selector = VideoFrameSelector('landing', test_folder)
            self.selector.imageUpdated.connect(self.load_media)
            self.selector.show()
        except Exception as e:
            print(f"选择落地帧时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"选择落地帧时出错: {str(e)}")

    def load_media(self):
        folder_name = self.sele_person_file
        
        try:
            # 使用当前选择的测试次数
            test_folder = os.path.join(folder_name, f'test_{self.current_test_number}')
            
            if os.path.exists(test_folder):
                self.jump_path = os.path.join(test_folder, "jump")
                self.speed_path = os.path.join(test_folder, "speed")
                
                print(f"加载测试文件夹: {test_folder}")  # 调试输出
                
                try:
                    # 从各个姿态文件夹加载图片
                    takeoff_dir = os.path.join(self.jump_path, 'takeoff')
                    flight_dir = os.path.join(self.jump_path, 'flight')
                    landing_dir = os.path.join(self.jump_path, 'landing')
                    
                    # 加载各个姿态的最佳匹配帧
                    if os.path.exists(takeoff_dir):
                        takeoff_img = os.path.join(takeoff_dir, 'compare_frame_best.jpg')
                        if os.path.exists(takeoff_img):
                            self.ui.label_img1.setPixmap(QPixmap(takeoff_img))
                            print(f"已加载起跳图片: {takeoff_img}")  # 调试输出
                    
                    if os.path.exists(flight_dir):
                        flight_img = os.path.join(flight_dir, 'compare_frame_best.jpg')
                        if os.path.exists(flight_img):
                            self.ui.label_img2.setPixmap(QPixmap(flight_img))
                            print(f"已加载空中图片: {flight_img}")  # 调试输出
                    
                    if os.path.exists(landing_dir):
                        landing_img = os.path.join(landing_dir, 'compare_frame_best.jpg')
                        if os.path.exists(landing_img):
                            self.ui.label_img3.setPixmap(QPixmap(landing_img))
                            print(f"已加载落地图片: {landing_img}")  # 调试输出
                    
                    # 加载速度图像
                    speed_plot_path = os.path.join(self.speed_path, 'speed_plot.png')
                    if os.path.exists(speed_plot_path):
                        self.ui.label_speedPlot.setPixmap(QPixmap(speed_plot_path))
                        print(f"已加载速度图像: {speed_plot_path}")  # 调试输出
                    else:
                        print(f"速度图像文件不存在: {speed_plot_path}")

                    # 加载分数
                    score_file = os.path.join(self.jump_path, 'jump_score_result.json')
                    if os.path.exists(score_file):
                        with open(score_file, 'r', encoding='utf-8') as file:
                            scores = json.load(file)
                            phase_scores = scores.get('phase_scores', {})
                            
                            # 显示各阶段分数
                            self.ui.labelTakeOffScore.setText(f"{phase_scores.get('takeoff', {}).get('score', 0):.2f}")
                            self.ui.labelHipExtensionScore.setText(f"{phase_scores.get('flight', {}).get('score', 0):.2f}")
                            self.ui.labelAbdominalContractionScore.setText(f"{phase_scores.get('landing', {}).get('score', 0):.2f}")
                            
                            # 显示总分
                            self.ui.labelAllScore.setText(f"{scores.get('overall_score', 0):.2f}")
                            print(f"已加载分数文件: {score_file}")  # 调试输出
                            
                            # 可以添加显示评价和建议的代码
                            if hasattr(self.ui, 'labelEvaluation'):  # 如果UI中有评价标签
                                self.ui.labelEvaluation.setText(scores.get('overall_evaluation', ''))
                            if hasattr(self.ui, 'textSuggestions'):  # 如果UI中有建议文本框
                                suggestions = '\n'.join(scores.get('suggestions', []))
                                self.ui.textSuggestions.setPlainText(suggestions)
                    else:
                        print(f"分数文件不存在: {score_file}")
                        self.ui.labelTakeOffScore.setText("N/A")
                        self.ui.labelHipExtensionScore.setText("N/A")
                        self.ui.labelAbdominalContractionScore.setText("N/A")
                        self.ui.labelAllScore.setText("N/A")
                    
                    # 加载 mean_vector.txt 文件中的速度
                    mean_vector_path = os.path.join(self.speed_path, 'mean_vector.txt')
                    if os.path.exists(mean_vector_path):
                        with open(mean_vector_path, 'r', encoding='utf-8') as file:
                            mean_speed = file.read().strip()  # 读取并去除空白字符
                            self.ui.labelMeanVector.setText(f"{mean_speed} m/s")  # 显示速度
                            print(f"已加载速度: {mean_speed} m/s")  # 调试输出
                    else:
                        print(f"速度文件不存在: {mean_vector_path}")

                    # 设置视频路径
                    video_file = os.path.join(self.jump_path, 'JumpVideo_analyzed.avi')
                    if os.path.exists(video_file):
                        self.setup_video_thread(self.jump_path, self.speed_path)
                        print(f"已加载视频: {video_file}")  # 调试输出
                    else:
                        print(f"视频文件不存在: {video_file}")
                    
                except Exception as e:
                    print(f"加载媒体文件时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"测试文件夹不存在: {test_folder}")
                
        except Exception as e:
            print(f"加载媒体总错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def setup_video_thread(self, jump_path, speed_path):
        video_path = os.path.join(jump_path if self.current_mode == 'jump' else speed_path, 'JumpVideo_analyzed.avi')
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        self.video_thread = VideoThread(video_path)
        self.video_thread.change_pixmap_signal.connect(self.ui.label_video.setPixmap)
        self.video_thread.start()
        # self.ui.StartBtn.setText('暂停')

    def prev_media(self):
        self.current_folder_index = max(0, self.current_folder_index - 1)
        self.load_media()

    def next_media(self):
        self.current_folder_index = min(len(self.folder_names) - 1, self.current_folder_index + 1)
        self.load_media()

    def switch_video_mode(self):
        self.current_mode = 'speed' if self.current_mode == 'jump' else 'jump'
        self.load_media()
        
    def toggle_pause_video(self):
        if not self.video_thread.isRunning():
            self.video_thread.start()
            # self.ui.StartBtn.setText('暂停')
        elif self.video_thread.paused:
            self.video_thread.resume()
            # self.ui.StartBtn.setText('暂停')
        else:
            self.video_thread.pause()
            # self.ui.StartBtn.setText('继续')
    def closeEvent(self, event):
        # 当关闭界面时调用
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()  # 等待线程完成
        event.accept()  # 关闭界面

    def setup_button_connections(self):
        """设置按钮连接"""
        try:
            # 按钮名称映射
            button_mappings = [
                ('Ca_img_but1', 'takeoff', '起跳姿态'),
                ('Ca_img_but2', 'flight', '空中姿态'),
                ('Ca_img_but3', 'landing', '落地姿态')
            ]
            
            for button_name, pose_type, pose_desc in button_mappings:
                # 检查按钮是否存在
                button = getattr(self.ui, button_name, None)
                if button is not None:
                    # 使用 partial 创建独立的处理函数
                    button.clicked.connect(partial(self.show_pose_details, pose_type))
                    print(f"已连接{pose_desc}按钮: {button_name}")
                else:
                    print(f"警告: 未找到按钮 {button_name}")
                    
        except Exception as e:
            print(f"设置按钮连接时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def show_pose_details(self, pose_type):
        """显示姿态匹配详细界面"""
        import traceback
        import pandas as pd
        
        print(f"触发按钮点击: {pose_type}")  # 调试输出
        try:
            if hasattr(self, 'jump_path'):
                pose_dir = os.path.join(self.jump_path, pose_type)
                print(f"检查文件夹: {pose_dir}")  # 调试输出
                
                if os.path.exists(pose_dir):
                    # 从Excel文件获取学生信息
                    try:
                        excel_file = os.path.join('student', 'student.xlsx')
                        if os.path.exists(excel_file):
                            # 读取Excel文件
                            df = pd.read_excel(excel_file)
                            
                            # 获取学生ID（从文件夹路径中提取）
                            student_id = os.path.basename(self.sele_person_file)
                            
                            # 确保student_id是字符串类型
                            df['student_id'] = df['student_id'].astype(str)
                            student_id = str(student_id)
                            
                            # 查找学生信息
                            student_info = df[df['student_id'] == student_id]
                            if not student_info.empty:
                                student_name = student_info.iloc[0]['name']
                                student_id = student_info.iloc[0]['student_id']
                            else:
                                student_name = "N/A"
                                student_id = "N/A"
                                print(f"在Excel中未找到学生: {student_id}")
                        else:
                            student_name = "N/A"
                            student_id = "N/A"
                            print(f"Excel文件不存在: {excel_file}")
                        
                        # 创建并保持对窗口的引用，传入学生信息
                        self.pose_details_windows[pose_type] = PoseMatchDetails(
                            pose_dir, 
                            pose_type,
                            student_name,
                            student_id
                        )
                        self.pose_details_windows[pose_type].show()
                        print(f"已创建并显示详细界面: {pose_type}")  # 调试输出
                    except Exception as e:
                        print(f"获取学生信息时出错: {str(e)}")
                        # 创建窗口但不传入学生信息
                        self.pose_details_windows[pose_type] = PoseMatchDetails(pose_dir, pose_type)
                        self.pose_details_windows[pose_type].show()
                else:
                    QMessageBox.warning(self, "错误", f"姿态文件夹不存在: {pose_dir}")
            else:
                QMessageBox.warning(self, "错误", "未找到跳远数据文件夹")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"显示详细界面时出错: {str(e)}")
            print(f"显示详细界面时出错: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    app = QApplication([])
    window = ShowResults()
    window.show()
    app.exec_()
