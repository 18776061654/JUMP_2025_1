import os
import cv2
from PySide6.QtGui import QImage, QPixmap  # 添加这行
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton  # 添加 QLabel 和 QPushButton
from PySide6.QtCore import QTimer  # 添加 QTimer
# 其他小部件方法，通用



# 获取速度和跳远的文件路径
def get_video_file_paths(student_manager):
    """获取视频文件路径"""
    # 使用当前选中的测试次数
    current_test = student_manager.main_window.current_test_number
    student_id = student_manager.main_window.sno_label.text()
    
    if not student_id:
        return None, None
        
    # 构建对应测试次数的文件夹路径
    student_folder = os.path.join(student_manager.results_folder, student_id)
    test_folder = os.path.join(student_folder, f'test_{current_test}')
    
    speed_folder = os.path.join(test_folder, 'speed')
    jump_folder = os.path.join(test_folder, 'jump')
    
    speed_file_name = os.path.join(speed_folder, "SpeedVideo.avi")
    jump_file_name = os.path.join(jump_folder, "JumpVideo.avi")
    
    return speed_file_name, jump_file_name


# 对帧进行等比例缩小
def convert_cv_qt( frame, width, height):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    
    # Calculate new size while maintaining aspect ratio
    ratio = min(width / w, height / h)
    new_width = int(w * ratio)
    new_height = int(h * ratio)

    return convert_to_Qt_format.scaled(new_width, new_height)







# ... 下面的代码是弹出的小窗口 ...

# 预览导入的跳远和速度视频
class VideoDialog(QDialog):
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Preview")
        self.setGeometry(100, 100, 640, 480)  # 调整窗口大小
        
        self.video_path = video_path
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.video_label = QLabel()
        self.layout.addWidget(self.video_label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.cap = cv2.VideoCapture(video_path)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            image = convert_cv_qt(frame, self.video_label.width(), self.video_label.height())
            self.video_label.setPixmap(QPixmap.fromImage(image))
        else:
            self.timer.stop()
            self.cap.release()
            self.close()

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        event.accept()


# 选择标定跑道的视频源
class CalibrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Options")
        self.setGeometry(100, 100, 300, 100)  # 调整窗口大小

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.video_source_button = QPushButton("选择的视频源")
        self.camera_button = QPushButton("摄像头")

        self.layout.addWidget(self.video_source_button)
        self.layout.addWidget(self.camera_button)

        # 添加按钮点击事件以关闭对话框
        self.video_source_button.clicked.connect(self.close)
        self.camera_button.clicked.connect(self.close)