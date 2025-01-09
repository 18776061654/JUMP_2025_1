from PySide6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QImage, QPixmap
import cv2
import numpy as np
from posture.JumpAnalyzer import JumpAnalyzer

class PoseDisplayThread(QThread):
    frame_ready = Signal(QPixmap)
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.running = True
        self.analyzer = JumpAnalyzer()

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # 使用 JumpAnalyzer 进行姿态检测和绘制
                frame_with_pose = self.analyzer.draw_pose_on_frame(frame)
                
                # 转换为Qt格式
                rgb_frame = cv2.cvtColor(frame_with_pose, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                
                # 发送处理后的帧
                self.frame_ready.emit(pixmap)
                
                # 控制显示速度
                self.msleep(30)  # 约30fps
            else:
                break
        cap.release()

    def stop(self):
        self.running = False

class PoseDisplayWindow(QMainWindow):
    def __init__(self, video_path):
        super().__init__()
        self.setWindowTitle("姿态检测实时显示")
        self.setMinimumSize(800, 600)
        
        # 创建主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 创建标签用于显示视频
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        
        # 创建并启动显示线程
        self.display_thread = PoseDisplayThread(video_path)
        self.display_thread.frame_ready.connect(self.update_frame)
        self.display_thread.start()

    def update_frame(self, pixmap):
        """更新显示的帧"""
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        """窗口关闭时停止线程"""
        self.display_thread.stop()
        self.display_thread.wait()
        event.accept() 