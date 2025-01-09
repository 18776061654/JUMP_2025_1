import sys
import cv2
import json
from PySide6.QtCore import QPoint, Qt, QSize
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog,QDialog,QMessageBox,QSlider,QLabel
from PySide6.QtGui import QPainter, QPixmap, QPen
from PySide6.QtUiTools import QUiLoader
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtCore import QTimer
from public.resources.newwindow.initposture_main import Ui_Main
from public.resources.newwindow.evaluation import Ui_Evaluation
from utils.model import load_model, process_frame
from utils.util import save_data
from utils.visual import draw_pic, KEYPOINT_CONNECTIONS
import numpy as np


# 重定义姿态
class EvaluationWindow(QDialog):
    def __init__(self,keypoint_path):
        super().__init__()
        
        self.keypoint_path = keypoint_path
        self.final_keypoints = []
        self.q_points = []
        self.index = None
       
        self.ui = Ui_Evaluation()
        self.ui.setupUi(self)
        
        self.ui.preFrame.clicked.connect(self.loadPreFrame)
        self.ui.nextFrame.clicked.connect(self.loadNextFrame)
        self.ui.save_button.clicked.connect(self.saveEvaluation)
        self.ui.editButton.clicked.connect(self.editEvaluation)

        self.scale_factor = 1.0  # 新增缩放因子，默认为1.0
        self.original_size = (self.ui.keypointPic.width(), self.ui.keypointPic.height())  # 原始图片大小
        self.scaled_width = int(self.original_size[0] * self.scale_factor)
        self.scaled_height = int(self.original_size[1] * self.scale_factor)
        
        self.is_drag_enabled = False  # 添加状态变量，默认为False
        self.ui.keypointPic.setMouseTracking(True)
        self.selected_point_index = None  # 新增，用于追踪当前选中的点索引
        
        self.ui.scaleSlider.valueChanged.connect(self.scaleImage)
        self.ui.scaleSlider.setMinimum(80) 
        self.ui.scaleSlider.setMaximum(150)  
        self.ui.scaleSlider.setValue(100)  
        self.ui.scaleSlider.setTickPosition(QSlider.TickPosition.TicksAbove)  # 设置刻度位置
        self.ui.scaleSlider.setTickInterval(5)  # 设置刻度间隔
        
        self.ui.scrollArea.setWidget(self.ui.keypointPic)
   
        if self.keypoint_path:
            with open(self.keypoint_path, 'r') as f:
                loaded_keypoints = json.load(f)   
            self.origin_keypoints = [list(map(tuple, kp)) for kp in loaded_keypoints]
            for index,keypoints in enumerate(self.origin_keypoints):
                if keypoints:
                    self.q_points = [(p[0], p[1]) for p in keypoints]
                    self.index = index
                    break     
        
            self.drawKeypoints()

    def mousePressEvent(self, event):
        # 将全局坐标转换为相对于 self.ui.keypointPic 的本地坐标
        local_pos = self.ui.keypointPic.mapFromGlobal(event.globalPosition().toPoint())
        click_threshold = 15 * self.scale_factor  # 根据缩放动态调整点击阈值
        for i, point in enumerate(self.q_points):
            # 根据归一化坐标计算点的实际位置
            scaled_point = QPoint(point[0] * self.scaled_width, point[1] * self.scaled_height)
            # 转换为相对于 keypointPic 的本地坐标
            if (scaled_point - local_pos).manhattanLength() < click_threshold:
                self.selected_point_index = i
                break
        print(self.selected_point_index)

    def mouseMoveEvent(self, event):
        if not self.is_drag_enabled or self.selected_point_index is None:
            return
        
        # 同样，转换坐标
        local_pos = self.ui.keypointPic.mapFromGlobal(event.globalPosition().toPoint())

        # 更新点位置
        new_x = local_pos.x() / self.scaled_width
        new_y = local_pos.y() / self.scaled_height
        self.q_points[self.selected_point_index] = (new_x, new_y)
        self.drawKeypoints()

    def mouseReleaseEvent(self, event):
        if not self.is_drag_enabled:
            return
        self.selected_point_index = None
        self.updateFinalKeypoints()  # 确保更新最终关键点
        
    def updateFinalKeypoints(self):
        # 将更新后的关键点坐标转换回原始比例并保存到self.final_keypoints
        self.final_keypoints = [[point[0], point[1]] for point in self.q_points]


    def scaleImage(self, value):
        self.scale_factor = value / 100.0  # 假设滑块范围是0-100，代表0%到100%的缩放
        self.scaled_width = int(self.original_size[0] * self.scale_factor)
        self.scaled_height = int(self.original_size[1] * self.scale_factor)
        self.drawKeypoints()

        
    def loadPreFrame(self):
        flag = True
        while flag:
            if self.index == 0:
                self.index = len(self.origin_keypoints) - 1
            self.index -= 1
            if self.origin_keypoints[self.index]:
                self.q_points = [(p[0], p[1]) for p in self.origin_keypoints[self.index]]
                self.drawKeypoints()
                flag = False
    
    def loadNextFrame(self):
        flag = True
        while flag:
            if self.index == len(self.origin_keypoints) - 1:
                self.index = 0
            self.index += 1
            if self.origin_keypoints[self.index]:
                self.q_points = [(p[0], p[1]) for p in self.origin_keypoints[self.index]]
                self.drawKeypoints()
                flag = False
        
    def drawKeypoints(self):
        if not hasattr(self, 'q_points') or not self.q_points:
            return  # 如果 self.q_points 未定义或为空，则直接返回
        
        scaled_size = QSize(self.scaled_width, self.scaled_height)
        pixmap = QPixmap(scaled_size)
        pixmap.fill(Qt.GlobalColor.white)

        painter = QPainter(pixmap)
        try:
            # 首先绘制连接线
            pen = QPen(Qt.GlobalColor.green, int(2 * self.scale_factor))  # 设置线条颜色为绿色，调整线条宽度
            painter.setPen(pen)
            
            for point1, point2 in KEYPOINT_CONNECTIONS:
                if point1 < len(self.q_points) and point2 < len(self.q_points):
                    x1, y1 = int(self.q_points[point1][0] * self.scaled_width), int(self.q_points[point1][1] * self.scaled_height)
                    x2, y2 = int(self.q_points[point2][0] * self.scaled_width), int(self.q_points[point2][1] * self.scaled_height)
                    painter.drawLine(x1, y1, x2, y2)

            # 然后绘制关键点
            pen.setColor(Qt.GlobalColor.red)  # 更改笔的颜色为红色
            pen.setWidth(int(5 * self.scale_factor))  # 调整点的大小
            painter.setPen(pen)

            for point in self.q_points:
                x = int(point[0] * self.scaled_width)
                y = int(point[1] * self.scaled_height)
                painter.drawPoint(x, y)

        finally:
            painter.end()

            
        self.ui.keypointPic.setPixmap(pixmap)
        self.ui.keypointPic.resize(scaled_size)

       
    def saveEvaluation(self):
        if not self.final_keypoints:
            self.final_keypoints = self.origin_keypoints[self.index]
        filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "JSON Files (*.json)")
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.final_keypoints, f)
                QMessageBox.information(self, "成功", "保存成功!")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存错误: {str(e)}")
        else:
            QMessageBox.warning(self, "警告", "未选择保存文件.")

    def editEvaluation(self):
        QMessageBox.information(self, "提示", "请手动拖拽点以进行姿态自定义")
        self.is_drag_enabled = True  # 新增，启用拖拽功能

class initposture_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.ui = Ui_Main()
        self.ui.setupUi(self)
        
        self.model = load_model()

        self.player1 = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player1.setVideoOutput(self.ui.originVideo)
        self.player1.setAudioOutput(self.audio_output)

        self.player2 = QMediaPlayer()
        self.player2.setVideoOutput(self.ui.keypointVideo)
        
        self.keypoint_path = 'keypoints.json'
        self.output_video_path = 'output.mp4'
        
        self.ui.controlButton.setVisible(False)
        self.ui.selectVideoButton.clicked.connect(self.openVideoFile)
        self.ui.controlButton.clicked.connect(self.handleVideoButton)
        self.ui.buildButton.clicked.connect(self.buildkeypointVideo)
        self.ui.editButton.clicked.connect(self.editEvaluation)
        
        self.player1.mediaStatusChanged.connect(self.handleMediaStatusChanged1)
        self.player2.mediaStatusChanged.connect(self.handleMediaStatusChanged2)
    
    def handleMediaStatusChanged1(self, status):
        # 检查媒体状态是否为播放结束
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player1.pause()  # 暂停播放器以保留最后一帧

    def handleMediaStatusChanged2(self, status):
        # 检查媒体状态是否为播放结束
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player2.pause()  # 暂停播放器以保留最后一帧

    def openVideoFile(self):
        video_name = QFileDialog.getOpenFileName()[0]
        
        if video_name:
            self.video_name = video_name 
        
        if self.video_name:
            self.ui.controlButton.setVisible(True)
            self.player1.setSource(self.video_name)
            self.player1.play()
        
    def handleVideoButton(self):
        button_text = self.ui.controlButton.text()
        if button_text == "播放":
            self.playVideo()
        elif button_text == "暂停":
            self.pauseVideo()
        
    def playVideo(self):
        self.player1.play()
        self.player2.play()
        self.ui.controlButton.setText("暂停")
 
    def pauseVideo(self):
        self.player1.pause()
        self.player2.pause()
        self.ui.controlButton.setText("播放")

        
    def buildkeypointVideo(self):
        self.player1.pause()
        self.ui.controlButton.setText("播放")
        
        keypoints = []

        # 保存关键点数据
        cap = cv2.VideoCapture(self.video_name)
        
        if not cap.isOpened():
            print("Error opening video file")
            return
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            points = process_frame(self.model, frame)
            keypoints.append(points)
            frame_count += 1
        cap.release()
        save_data(keypoints, self.keypoint_path)
        print('done')
        
        with open(self.keypoint_path, 'r') as f:
            loaded_keypoints = json.load(f)

        # 将读取的keypoints转换回原始格式（即将内部的列表转换为元组，以匹配原始keypoints的格式）
        original_format_keypoints = [list(map(tuple, kp)) for kp in loaded_keypoints]

        # 增大画布尺寸
        canvas_width = self.ui.keypointVideo.width() * 3
        canvas_height = self.ui.keypointVideo.height() * 2

        # 创建一个透明图像用于绘制轨迹
        trajectory_frame = np.zeros((canvas_height, canvas_width, 4), np.uint8)

        # 优化线条和点的绘制，每隔五帧画一个姿态
        offset_x = 0
        for index, keypoints in enumerate(original_format_keypoints):
            if index % 5 == 0:  # 每隔五帧处理一次
                for point1, point2 in KEYPOINT_CONNECTIONS:
                    if point1 < len(keypoints) and point2 < len(keypoints):
                        x1, y1 = int((keypoints[point1][0] * self.ui.keypointVideo.width()) + offset_x), int(keypoints[point1][1] * self.ui.keypointVideo.height())
                        x2, y2 = int((keypoints[point2][0] * self.ui.keypointVideo.width()) + offset_x), int(keypoints[point2][1] * self.ui.keypointVideo.height())
                        cv2.line(trajectory_frame, (x1, y1), (x2, y2), (0, 255, 0, 255), 2, cv2.LINE_AA)  # 绿色抗锯齿线条，确保不透明

                for point in keypoints:
                    x = int((point[0] * self.ui.keypointVideo.width()) + offset_x)
                    y = int(point[1] * self.ui.keypointVideo.height())
                    cv2.circle(trajectory_frame, (x, y), 5, (0, 0, 255, 255), -1, cv2.LINE_AA)  # 红色抗锯齿点，确保不透明
                
                # 每绘制一组姿态，增加偏移量
                offset_x += self.ui.keypointVideo.width() * 0.5  # 可调整的偏移量，避免姿态重叠

        # 将轨迹图保存为图像文件
        output_image_path = 'trajectory_output.png'
        cv2.imwrite(output_image_path, trajectory_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f'Trajectory image saved to {output_image_path}')

        # 显示轨迹图
        self.displayTrajectoryImage(output_image_path)

    def displayTrajectoryImage(self, image_path):
        # 创建一个 QLabel 用于显示轨迹图像
        trajectory_label = QLabel(self)
        pixmap = QPixmap(image_path)
        trajectory_label.setPixmap(pixmap)
        trajectory_label.resize(pixmap.size())
        trajectory_label.show()



    
    def editEvaluation(self):
        # 读取关键点数据
        import os
        if not os.path.exists(self.keypoint_path):
            QMessageBox.critical(self, "Error", "未生成点图，先选择视频生成点图")
            return
        self.evaluation_window = EvaluationWindow(self.keypoint_path)
        self.evaluation_window.resize(1600, 900)
        self.evaluation_window.show()
                
def get_duration_from_cv2(filename):
        cap = cv2.VideoCapture(filename)
        if cap.isOpened():
            rate = cap.get(5)
            frame_num =cap.get(7)
            duration = frame_num/rate
            cap.release()
            return duration
        return -1
            

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = initposture_MainWindow()
    window.resize(1600, 900)
    # Show the main window
    window.show()

    # Start the event loop
    sys.exit(app.exec())