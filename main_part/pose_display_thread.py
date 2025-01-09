from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage, QPixmap
import cv2
import numpy as np
from posture.JumpAnalyzer import JumpAnalyzer
import os

class PoseDisplayThread(QThread):
    frame_ready = Signal(QPixmap)
    
    def __init__(self, video_path, student_id, save_video=True):
        super().__init__()
        self.video_path = video_path
        self.student_id = student_id
        self.running = True
        self.analyzer = JumpAnalyzer()
        
        # 初始化视频写入器
        self.save_video = save_video
        if save_video:
            output_path = os.path.join(os.path.dirname(video_path), 'JumpVideo_analyzed.avi')
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            cap.release()

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # 使用 JumpAnalyzer 的 process_frame_for_display 方法
                    frame_with_pose = self.analyzer.process_frame_for_display(frame)
                    
                    # 保存处理后的帧
                    if self.save_video:
                        self.out.write(frame_with_pose)
                    
                    # 转换为Qt格式并发送显示
                    rgb_frame = cv2.cvtColor(frame_with_pose, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    self.frame_ready.emit(pixmap)
                    
                    # 控制显示速度
                    self.msleep(30)  # 约30fps
                else:
                    # 视频结束，重新开始播放
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
            cap.release()
            if self.save_video:
                self.out.release()
                
        except Exception as e:
            print(f"处理视频时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def stop(self):
        self.running = False 