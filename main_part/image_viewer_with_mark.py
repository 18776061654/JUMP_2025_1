import sys
import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout,QMessageBox,QInputDialog
import public.utils.font_util as font_util
import configparser
import math
# 标定线部分
class ImageContainer(QWidget):
    def __init__(self, image, text):
        super().__init__()

        self.points = []
        self.g_zoom = 1
        self.g_step = 0.1
        self.distance_input_complete = True
        self.real_distance_1 = 0  # 保存第一条线的实际距离
        self.real_distance_2 = 0  # 保存第二条线的实际距离

        # 调整图像大小
        container_width = 1600
        container_height = 800
        image_resized = cv2.resize(image, (container_width, container_height), interpolation=cv2.INTER_AREA)

        font_size = int(min(image_resized.shape[1], image_resized.shape[0]) * 0.025)
        self.image_clean = image_resized.copy()  # 保存一个未标记的原始图像副本
        self.image_original = font_util.cv2_chinese_text(self.image_clean.copy(), text, (20, 30), (255, 0, 0), font_size)
        self.image_zoom = self.image_original.copy()
        self.image_show = self.image_original.copy()

        self.g_location_win = [0, 0]
        self.location_win = [0, 0]
        self.g_location_click = [0, 0]
        self.g_location_release = [0, 0]

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        # 调整初始缩放比例并更新视图
        self.adjust_initial_zoom()
        self.update_image_display()

    def adjust_initial_zoom(self):
        """根据窗口和图像大小调整初始缩放比例"""
        w1, h1 = self.image_original.shape[1], self.image_original.shape[0]  # 图像大小
        w2, h2 = self.size().width(), self.size().height()  # 窗口大小
        self.g_zoom = min(w2 / w1, h2 / h1)  # 根据窗口大小调整初始缩放比例

        # 使用新的缩放比例更新图像
        self.image_zoom = cv2.resize(self.image_original, 
                                     (int(w1 * self.g_zoom), int(h1 * self.g_zoom)), 
                                     interpolation=cv2.INTER_AREA)
        self.update_view()

    def resizeEvent(self, event):
        """在窗口大小变化时更新视图"""
        self.adjust_initial_zoom()
        super().resizeEvent(event)

    def update_image_display(self):
        height, width, channel = self.image_show.shape
        bytes_per_line = 3 * width
        contiguous_image = np.ascontiguousarray(self.image_show)
        q_image = QImage(contiguous_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_view(self):
        w1, h1 = self.image_zoom.shape[1], self.image_zoom.shape[0]
        w2, h2 = self.label.size().width(), self.label.size().height()
        show_wh = [min(w1, w2), min(h1, h2)]

        # 确保视图位置的合法性
        self.check_location([w1, h1], [w2, h2], self.g_location_win)

        self.image_show = self.image_zoom[self.g_location_win[1]:self.g_location_win[1] + show_wh[1], 
                                          self.g_location_win[0]:self.g_location_win[0] + show_wh[0]]
        self.update_image_display()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            # 计算当前鼠标的位置，拖动图像
            self.g_location_release = [int(event.position().x()), int(event.position().y())]

            h1, w1 = self.image_zoom.shape[0:2]
            w2, h2 = self.label.size().width(), self.label.size().height()
            show_wh = [0, 0]

            if w1 < w2 and h1 < h2:
                show_wh = [w1, h1]
                self.g_location_win = [0, 0]
            elif w1 >= w2 and h1 < h2:
                show_wh = [w2, h1]
                self.g_location_win[0] = max(0, min(w1 - w2, self.location_win[0] + self.g_location_click[0] - self.g_location_release[0]))
            elif w1 < w2 and h1 >= h2:
                show_wh = [w1, h2]
                self.g_location_win[1] = max(0, min(h1 - h2, self.location_win[1] + self.g_location_click[1] - self.g_location_release[1]))
            else:
                show_wh = [w2, h2]
                self.g_location_win[0] = max(0, min(w1 - w2, self.location_win[0] + self.g_location_click[0] - self.g_location_release[0]))
                self.g_location_win[1] = max(0, min(h1 - h2, self.location_win[1] + self.g_location_click[1] - self.g_location_release[1]))

            # 更新显示图像
            self.image_show = self.image_zoom[int(self.g_location_win[1]):int(self.g_location_win[1]) + int(show_wh[1]), 
                                            int(self.g_location_win[0]):int(self.g_location_win[0]) + int(show_wh[0])]
            self.update_image_display()
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 用于拖拽的鼠标点击位置，不做偏移处理
            self.g_location_click = [int(event.position().x()), int(event.position().y())]
            self.location_win = self.g_location_win.copy()

        elif event.button() == Qt.RightButton:
            if not self.distance_input_complete:
                # 如果距离输入未完成，则提示用户需要输入距离
                QMessageBox.warning(self, "提示", "请先输入距离并确认。")
                return

            # 用于标记的鼠标点击位置，应用偏移
            arrow_offset_x = 8  # 根据鼠标箭头的大小调整
            arrow_offset_y = 8

            # 获取鼠标点击在窗口中的位置
            window_x = int(event.position().x()) - arrow_offset_x
            window_y = int(event.position().y()) - arrow_offset_y

            # 获取label尺寸
            label_width = self.label.width()
            label_height = self.label.height()

            # 获取显示的pixmap
            pixmap = self.label.pixmap()
            if pixmap:
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
            else:
                pixmap_width, pixmap_height = 0, 0

            # 计算偏移量
            offset_x = (label_width - pixmap_width) // 2
            offset_y = (label_height - pixmap_height) // 2

            # 确保鼠标点击在有效的pixmap区域内
            if offset_x <= window_x <= offset_x + pixmap_width and offset_y <= window_y <= offset_y + pixmap_height:
                # 将窗口坐标映射到显示图像的坐标
                image_x = int((window_x - offset_x + self.g_location_win[0]) / self.g_zoom)
                image_y = int((window_y - offset_y + self.g_location_win[1]) / self.g_zoom)

                # 限制标记点数量不超过4个
                if len(self.points) < 4:
                    self.points.append((image_x, image_y))

                    # 在图像上绘制标记点
                    cv2.circle(self.image_original, (image_x, image_y), 5, (0, 0, 255), -1)

                    # 在标记点旁边显示顺序数字
                    cv2.putText(self.image_original, str(len(self.points)), (image_x + 10, image_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # 如果已经标记了2个点，连接第1和第2个点
                    if len(self.points) == 2:
                        cv2.line(self.image_original, self.points[0], self.points[1], (0, 255, 0), 2)
                        

                        # 更新缩放后的图像
                        self.image_zoom = cv2.resize(self.image_original,
                                                    (int(self.image_original.shape[1] * self.g_zoom),
                                                    int(self.image_original.shape[0] * self.g_zoom)),
                                                    interpolation=cv2.INTER_AREA)
                        self.update_view()

                        # 弹出输入框，要求用户输入第1条线的实际距离
                        distance, ok = QInputDialog.getDouble(self, "输入距离", "请输入这条线的实际距离:", minValue=0.0)
                        if ok:
                            self.real_distance_1 = distance
                            self.distance_input_complete = True  # 允许继续标记

                            # 计算连线的中点
                            mid_point_x = (self.points[0][0] + self.points[1][0]) // 2
                            mid_point_y = (self.points[0][1] + self.points[1][1]) // 2

                            # 在中点的上方显示距离
                            text_position = (mid_point_x, mid_point_y - 10)  # 距离显示在中点上方
                            cv2.putText(self.image_original, f"{distance:.2f}m", text_position,
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        else:
                            # 如果用户取消输入，移除第二个点
                            self.points.pop()
                            self.distance_input_complete = True

                    # 如果已经标记了4个点，连接第3和第4个点
                    elif len(self.points) == 4:
                        cv2.line(self.image_original, self.points[2], self.points[3], (0, 255, 0), 2)
                        
                        # 更新缩放后的图像
                        self.image_zoom = cv2.resize(self.image_original,
                                                    (int(self.image_original.shape[1] * self.g_zoom),
                                                    int(self.image_original.shape[0] * self.g_zoom)),
                                                    interpolation=cv2.INTER_AREA)
                        self.update_view()                        
                        # 弹出输入框，要求用户输入第2条线的实际距离
                        distance, ok = QInputDialog.getDouble(self, "输入距离", "请输入这条线的实际距离:", minValue=0.0)
                        if ok:
                            self.real_distance_2 = distance
                            self.distance_input_complete = True  # 允许继续标记

                            # 计算连线的中点
                            mid_point_x = (self.points[2][0] + self.points[3][0]) // 2
                            mid_point_y = (self.points[2][1] + self.points[3][1]) // 2

                            # 在中点的上方显示距离
                            text_position = (mid_point_x, mid_point_y - 10)  # 距离显示在中点上方
                            cv2.putText(self.image_original, f"{distance:.2f}m", text_position,
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        else:
                            # 如果用户取消输入，移除第四个点
                            self.points.pop()
                            self.distance_input_complete = True

                    # 更新缩放后的图像
                    self.image_zoom = cv2.resize(self.image_original,
                                                (int(self.image_original.shape[1] * self.g_zoom),
                                                int(self.image_original.shape[0] * self.g_zoom)),
                                                interpolation=cv2.INTER_AREA)
                    self.update_view()

                    # 打印标记点信息到控制台
                    print(f"Point {len(self.points)}: ({image_x}, {image_y})")
                    print("Marked Points:", self.points)

                else:
                    # 如果已经标记了4个点，弹出提示框
                    QMessageBox.warning(self, "提示", "已经标记了四个点，无法继续标记。")



    def wheelEvent(self, event):
        z = self.g_zoom
        self.g_zoom = self.count_zoom(event.angleDelta().y(), self.g_step, self.g_zoom)
        w1, h1 = int(self.image_original.shape[1] * self.g_zoom), int(self.image_original.shape[0] * self.g_zoom)

        # 记录当前鼠标位置相对窗口的位置，用于放大缩小后调整视图位置
        pos = event.position()

        self.image_zoom = cv2.resize(self.image_original, (w1, h1), interpolation=cv2.INTER_AREA)

        self.g_location_win = [int((self.g_location_win[0] + pos.x()) * self.g_zoom / z - pos.x()),
                               int((self.g_location_win[1] + pos.y()) * self.g_zoom / z - pos.y())]

        self.update_view()

    def check_location(self, img_wh, win_wh, win_xy):
        for i in range(2):
            if win_xy[i] < 0:
                win_xy[i] = 0
            elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] > win_wh[i]:
                win_xy[i] = img_wh[i] - win_wh[i]
            elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] < win_wh[i]:
                win_xy[i] = 0

    def count_zoom(self, flag, step, zoom):
        if flag > 0:
            zoom += step
            if zoom > 1 + step * 20:
                zoom = 1 + step * 20
        else:
            zoom -= step
            if zoom <= 1:  # 设置一个最小的缩放比例
                zoom = 1
        zoom = round(zoom, 2)
        return zoom

    def clear_points(self):
        # 清除所有已标记的点
        self.points.clear()

        # 恢复为原始未标记的图像
        self.image_original = self.image_clean.copy()

        # 根据当前缩放比例调整显示的图像
        self.image_zoom = cv2.resize(self.image_original, 
                                     (int(self.image_original.shape[1] * self.g_zoom), 
                                      int(self.image_original.shape[0] * self.g_zoom)),
                                     interpolation=cv2.INTER_AREA)

        # 更新视图
        self.update_view()


    def get_marked_points(self):
    # 返回已标记的点
        return self.points

class ImageWindow(QMainWindow):
    def __init__(self, image):
        super().__init__()

        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 1600, 900)

        self.image_container = ImageContainer(image, "标记点")

        # 添加“确定”和“重新标记”按钮
        self.ok_button = QPushButton("确定")
        self.reset_button = QPushButton("重新标记")
        self.reset_button.clicked.connect(self.reset_points)
        self.ok_button.clicked.connect(self.return_marked_points)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_container)
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def reset_points(self):
        self.image_container.clear_points()


    def return_marked_points(self):
        marked_points = self.image_container.get_marked_points()
        if len(marked_points) < 4:
            QMessageBox.warning(self, "提示", "请标记四个点后再点击确定。")
            return

        # 获取实际距离
        real_distance_1 = self.image_container.real_distance_1
        real_distance_2 = self.image_container.real_distance_2


        # 读取现有的配置文件
        config = configparser.ConfigParser()
        config.read('config/config.ini')  # 假设配置文件名为 config.ini

        # 更新 [Runway] 部分
        if 'Runway' not in config:
            config.add_section('Runway')

        config['Runway']['real_distance_1'] = str(real_distance_1)
        config['Runway']['pixel_point1_1'] = f"({marked_points[0][0]}, {marked_points[0][1]})"
        config['Runway']['pixel_point2_1'] = f"({marked_points[1][0]}, {marked_points[1][1]})"

        config['Runway']['real_distance_2'] = str(real_distance_2)
        config['Runway']['pixel_point1_2'] = f"({marked_points[2][0]}, {marked_points[2][1]})"
        config['Runway']['pixel_point2_2'] = f"({marked_points[3][0]}, {marked_points[3][1]})"

        # 将更新内容写回配置文件，保留原有的部分
        with open('config/config.ini', 'w') as configfile:
            config.write(configfile)

        print("Marked Points and distances saved to config.ini.")
        self.close()

