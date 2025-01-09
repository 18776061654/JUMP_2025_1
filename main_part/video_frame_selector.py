from PySide6.QtWidgets import QMainWindow, QSlider, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QMessageBox
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
import cv2
import os
import json
from posture.key_score import KeyScoreCalculator

class VideoFrameSelector(QMainWindow):
    imageUpdated = Signal()  # 当新的帧被选择时发出信号

    def __init__(self, pose_type, student_folder):
        """
        初始化视频帧选择器
        Args:
            pose_type: 姿态类型 ('takeoff', 'flight', 'landing')
            student_folder: 学生文件夹路径 (如 .../test_1/jump/takeoff)
        """
        super().__init__()
        self.pose_type = pose_type
        self.student_folder = student_folder
        
        # 构建正确的视频路径
        # 从 .../test_1/jump/takeoff 回退两级到 .../test_1
        test_folder = os.path.dirname(os.path.dirname(student_folder))
        self.video_path = os.path.join(test_folder, 'jump', 'JumpVideo.avi')
        print(f"视频路径: {self.video_path}")  # 调试输出
        
        self.setup_ui()
        if not self.load_video():
            print("视频加载失败")

    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle(f"选择{self.pose_type}阶段帧")
        self.setMinimumSize(800, 600)

        # 创建主窗口部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 创建视频显示标签
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # 创建滑动条
        slider_layout = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        slider_layout.addWidget(self.frame_slider)
        layout.addLayout(slider_layout)

        # 创建帧号显示标签
        self.frame_label = QLabel("Frame: 0")
        layout.addWidget(self.frame_label)

        # 创建按钮布局
        button_layout = QHBoxLayout()
        
        # 上一帧按钮
        self.prev_button = QPushButton("上一帧")
        self.prev_button.clicked.connect(self.prev_frame)
        button_layout.addWidget(self.prev_button)
        
        # 下一帧按钮
        self.next_button = QPushButton("下一帧")
        self.next_button.clicked.connect(self.next_frame)
        button_layout.addWidget(self.next_button)
        
        # 选择按钮
        self.select_button = QPushButton("选择此帧")
        self.select_button.clicked.connect(self.select_frame)
        button_layout.addWidget(self.select_button)
        
        layout.addLayout(button_layout)

    def load_video(self):
        """加载视频文件"""
        try:
            # 检查视频文件是否存在
            if not os.path.exists(self.video_path):
                print(f"视频文件不存在: {self.video_path}")
                QMessageBox.warning(self, "错误", f"视频文件不存在: {self.video_path}")
                self.close()
                return False

            # 尝试打开视频
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"无法打开视频文件: {self.video_path}")
                QMessageBox.warning(self, "错误", "无法打开视频文件")
                self.close()
                return False

            # 获取视频信息
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                print("视频帧数为0")
                QMessageBox.warning(self, "错误", "视频文件可能损坏")
                self.close()
                return False

            print(f"成功加载视频，总帧数: {total_frames}")
            self.frame_slider.setRange(0, total_frames - 1)

            # 读取第一帧
            ret = self.show_frame(0)
            if not ret:
                print("无法读取视频第一帧")
                QMessageBox.warning(self, "错误", "无法读取视频帧")
                self.close()
                return False

            return True

        except Exception as e:
            print(f"加载视频时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载视频时出错: {str(e)}")
            self.close()
            return False

    def show_frame(self, frame_number):
        """显示指定帧"""
        try:
            if not hasattr(self, 'cap') or not self.cap.isOpened():
                return False

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if not ret:
                print(f"无法读取帧 {frame_number}")
                return False

            # 转换颜色空间
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            # 转换为QImage
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 调整大小以适应标签
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            self.frame_label.setText(f"Frame: {frame_number}")
            self.current_frame = frame
            return True

        except Exception as e:
            print(f"显示帧时出错: {str(e)}")
            return False

    def on_slider_changed(self, value):
        """滑动条值改变时的处理函数"""
        self.show_frame(value)

    def prev_frame(self):
        """显示上一帧"""
        current = self.frame_slider.value()
        if current > 0:
            self.frame_slider.setValue(current - 1)

    def next_frame(self):
        """显示下一帧"""
        current = self.frame_slider.value()
        if current < self.frame_slider.maximum():
            self.frame_slider.setValue(current + 1)

    def select_frame(self):
        """选择当前帧作为匹配帧"""
        try:
            # 创建姿态文件夹
            pose_folder = os.path.join(self.student_folder)
            os.makedirs(pose_folder, exist_ok=True)

            # 1. 首先从 jump_analysis.json 获取选中帧的bbox数据
            selected_frame = self.frame_slider.value()
            analysis_file = os.path.join(os.path.dirname(self.student_folder), 'jump_analysis.json')
            
            if os.path.exists(analysis_file):
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                    
                # 找到选中帧的数据
                selected_frame_data = None
                for frame_data in analysis_data['data']['frames']:
                    if frame_data['frame_idx'] == selected_frame:
                        selected_frame_data = frame_data
                        break

                if selected_frame_data and 'bbox' in selected_frame_data:
                    # 获取边界框坐标
                    bbox = selected_frame_data['bbox']
                    x1, y1, x2, y2 = map(int, bbox)  # 确保坐标是整数

                    # 添加一些边距
                    margin = 20
                    height, width = self.current_frame.shape[:2]
                    
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(width, x2 + margin)
                    y2 = min(height, y2 + margin)

                    # 裁剪图像
                    cropped_frame = self.current_frame[y1:y2, x1:x2]
                else:
                    print("警告: 未找到边界框数据，使用原始帧")
                    cropped_frame = self.current_frame
            else:
                print("警告: 未找到分析文件，使用原始帧")
                cropped_frame = self.current_frame

            # 保存原始帧和裁剪帧
            frame_path = os.path.join(pose_folder, 'compare_frame_best.jpg')
            frame_cropped_path = os.path.join(pose_folder, 'compare_frame_best_cropped.jpg')
            
            cv2.imwrite(frame_path, self.current_frame)
            cv2.imwrite(frame_cropped_path, cropped_frame)

            # 更新姿态匹配结果
            self.update_pose_match_result()

            # 重新计算得分
            self.recalculate_scores()

            # 发送更新信号
            self.imageUpdated.emit()
            
            # 关闭窗口
            self.close()

        except Exception as e:
            print(f"选择帧时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_pose_match_result(self):
        """更新姿态匹配结果文件"""
        try:
            selected_frame = self.frame_slider.value()
            
            # 1. 首先从 jump_analysis.json 获取选中帧的关键点数据
            analysis_file = os.path.join(os.path.dirname(self.student_folder), 'jump_analysis.json')
            if os.path.exists(analysis_file):
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                    
                # 在所有帧中找到选中的帧
                selected_frame_data = None
                for frame_data in analysis_data['data']['frames']:
                    if frame_data['frame_idx'] == selected_frame:
                        selected_frame_data = frame_data
                        break
                
                if selected_frame_data:
                    # 2. 更新 pose_matches.json
                    pose_matches_file = os.path.join(self.student_folder, 'pose_matches.json')
                    if os.path.exists(pose_matches_file):
                        with open(pose_matches_file, 'r', encoding='utf-8') as f:
                            matches_data = json.load(f)
                        
                        # 更新第一个匹配的数据（最佳匹配）
                        if matches_data['matches']:
                            matches_data['matches'][0].update({
                                'frame_idx': selected_frame,
                                'keypoints': selected_frame_data['keypoints'],
                                'scores': selected_frame_data['scores'],
                                'bbox': selected_frame_data['bbox']
                            })
                        
                        with open(pose_matches_file, 'w', encoding='utf-8') as f:
                            json.dump(matches_data, f, indent=4, ensure_ascii=False)
                    
                    # 3. 更新姿态得分文件
                    pose_score_file = os.path.join(self.student_folder, f'{self.pose_type}_score_result.json')
                    if os.path.exists(pose_score_file):
                        with open(pose_score_file, 'r', encoding='utf-8') as f:
                            score_data = json.load(f)
                        
                        # 更新选中的帧号
                        score_data['selected_frame'] = selected_frame
                        
                        # 更新关键点数据
                        if 'frame_data' in score_data:
                            score_data['frame_data'] = {
                                'keypoints': selected_frame_data['keypoints'],
                                'scores': selected_frame_data['scores'],
                                'bbox': selected_frame_data['bbox']
                            }
                        
                        with open(pose_score_file, 'w', encoding='utf-8') as f:
                            json.dump(score_data, f, indent=4, ensure_ascii=False)
                else:
                    print(f"警告: 在分析数据中未找到帧 {selected_frame}")
            else:
                print(f"警告: 未找到分析文件 {analysis_file}")

        except Exception as e:
            print(f"更新姿态匹配结果时出错: {str(e)}")
            traceback.print_exc()

    def recalculate_scores(self):
        """重新计算得���"""
        try:
            calculator = KeyScoreCalculator()
            # 获取jump文件夹路径（从姿态文件夹回退一级）
            jump_folder = os.path.dirname(self.student_folder)
            
            # 重新计算所有得分
            scores = calculator.calculate_all_scores(jump_folder)
            
            # 生成新的跳远评分文件
            calculator.generate_jump_score(jump_folder)
            
            print("得分重新计算完成")
            print("各阶段得分:", scores)

        except Exception as e:
            print(f"重新计算得分时出错: {str(e)}")
            traceback.print_exc()

    def closeEvent(self, event):
        """窗口关闭时释放资源"""
        if hasattr(self, 'cap'):
            self.cap.release()
        super().closeEvent(event) 