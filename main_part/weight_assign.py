from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QTableView,
    QPushButton, QVBoxLayout, QHBoxLayout, QWidget
)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QAbstractTableModel, Qt, QEvent
from PySide6.QtGui import QStandardItemModel, QStandardItem, QImage, QPixmap
import json
import sys
import numpy as np
import cv2
import os

# 在创建 QApplication 之前设置属性
QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

human_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Human"))
sys.path.append(human_path)
from Human.Human import Human

def normalize_points(points, canvas_width, canvas_height, scale_factor=0.8):
    """
    将点数组归一化并居中，允许调整缩放比例
    Args:
        points (list): 原始点数组
        canvas_width (int): 画布宽度
        canvas_height (int): 画布高度
        scale_factor (float): 缩放因子，决定点数组占画布的比例（0到1之间，默认0.8）

    Returns:
        list: 归一化并居中的点数组
    """
    # 获取点的最小值和最大值
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # 计算点数组的宽度和高度
    points_width = max_x - min_x
    points_height = max_y - min_y

    # 缩放比例，保持纵横比，并应用缩放因子
    scale = min(canvas_width / points_width, canvas_height / points_height) * scale_factor

    # 缩放和居中点
    normalized_points = []
    for x, y in points:
        norm_x = (x - min_x) * scale + (canvas_width - points_width * scale) / 2
        norm_y = (y - min_y) * scale + (canvas_height - points_height * scale) / 2
        normalized_points.append((norm_x, norm_y))

    return normalized_points


def draw_points_on_canvas(points, human, canvas_width, canvas_height):
    """
    在画布上绘制点
    Args:
        points (list): 点数组
        canvas_width (int): 画布宽度
        canvas_height (int): 画布高度

    Returns:
        np.ndarray: 绘制完成的画布图像
    """
    # 创建空白画布
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    point_scores = [1 for _ in points] 
    point_instance = [{
        "keypoints": points,
        "keypoint_scores":point_scores,
    }]

    # 调用 plot_construct_point 方法绘制结构
    canvas = human.draw_pose(canvas, point_instance)

    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGB)
    return canvas_rgb

class WeightApp(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            # 加载 UI 文件
            ui_file = QFile("public/resources/newwindow/weight.ui")
            ui_file.open(QFile.ReadOnly)

            loader = QUiLoader()
            self.ui = loader.load(ui_file)
            ui_file.close()

            # 创建一个中央部件和布局
            central_widget = QWidget()
            main_layout = QVBoxLayout(central_widget)
            main_layout.setSpacing(5)  # 减小控件间距
            main_layout.setContentsMargins(5, 5, 5, 5)  # 减小边距
            
            # 创建按钮布局
            button_layout = QHBoxLayout()
            button_layout.setSpacing(5)  # 减小按钮间距
            
            # 将原有的按钮添加到布局中并连接信号
            self.ui.choseButton.clicked.connect(self.on_chose_clicked)
            self.ui.addAngleButton.clicked.connect(self.on_add_angle_clicked)
            self.ui.saveButton.clicked.connect(self.on_save_clicked)
            
            button_layout.addWidget(self.ui.choseButton)
            button_layout.addWidget(self.ui.addAngleButton)
            button_layout.addWidget(self.ui.saveButton)
            
            # 创建并添加删除按钮
            self.delete_button = QPushButton("删除选中行")
            self.delete_button.clicked.connect(self.on_delete_clicked)
            button_layout.addWidget(self.delete_button)
            
            # 创建水平布局来放置姿态显示和表格
            content_layout = QHBoxLayout()
            content_layout.setSpacing(10)  # 设置间距
            
            # 添加姿态显示区域（左侧）
            content_layout.addWidget(self.ui.posture, 2)  # 比例为2
            
            # 添加表格（右侧）
            content_layout.addWidget(self.ui.tableView, 1)  # 比例为1
            
            # 将布局添加到主布局
            main_layout.addLayout(button_layout)
            main_layout.addLayout(content_layout)
            
            # 设置中央部件
            self.setCentralWidget(central_widget)

            # 调整窗口大小和表格列宽
            self.setFixedSize(1000, 550)
            
            # 添加模型到 tableView
            self.table_model = QStandardItemModel()
            self.table_model.setHorizontalHeaderLabels(["关键角", "权重", "推荐角度"])
            self.ui.tableView.setModel(self.table_model)
            
            # 设置表格列宽
            self.ui.tableView.setColumnWidth(0, 100)  # 关键角列
            self.ui.tableView.setColumnWidth(1, 60)   # 权重列
            self.ui.tableView.setColumnWidth(2, 60)   # 推荐角度列

            # 捕捉鼠标点击事件
            self.ui.posture.installEventFilter(self)

            # 初始化 Human 类
            self.human = Human(config_path='config/config.ini')

            # 初始化变量
            self.marked_points = []
            self.add_button_count = 0
            self.coordinates = None
            self.canvas = None

        except Exception as e:
            print(f"加载 UI 文件时出错: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def eventFilter(self, source, event):
        if (source == self.ui.posture and 
            event.type() == QEvent.MouseButtonPress):
            self.on_canvas_clicked(event)
            return True
        return super().eventFilter(source, event)

    def on_chose_clicked(self):
        """选择 JSON 文件并验证并绘制点"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 JSON 文件", "", "JSON Files (*.json)"
        )

        if not file_path:
            return  # 用户取消操作

        try:
            # 打开并解析 JSON 文件
            with open(file_path, "r") as file:
                data = json.load(file)

            # 验证 JSON 文件中的 keypoints �� coordinates 字段
            if "keypoints" not in data or "coordinates" not in data["keypoints"]:
                raise ValueError("JSON 文件中缺少 keypoints 或 coordinates 字段！")

            # 存点数组
            self.coordinates = data["keypoints"]["coordinates"]

            # 获取画布尺寸
            self.canvas_width = self.ui.posture.width()
            self.canvas_height = self.ui.posture.height()

            # 归一化并居中点数组
            self.coordinates = normalize_points(self.coordinates, self.canvas_width, self.canvas_height)

            # 绘制点数组到画布
            self.canvas = draw_points_on_canvas(self.coordinates, self.human, self.canvas_width, self.canvas_height)

            canvas = self.canvas.copy()
            # 将画布转换为 QPixmap 并显示在 QLabel（posture）控件中
            height, width, channel = canvas.shape
            bytes_per_line = channel * width
            q_image = QImage(canvas.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.ui.posture.setPixmap(pixmap)

            QMessageBox.information(self, "成功", "JSON 文件加载成功并绘制！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载 JSON 文件失败: {e}")

    def on_canvas_clicked(self, event):
        """处理画布点击事件"""
        if not self.coordinates:
            return
        
        x, y = event.pos().x(), event.pos().y()
        
        # 查找最近的点
        for idx, (px, py) in enumerate(self.coordinates):
            if abs(px - x) <= 15 and abs(py - y) <= 15:  # 增加点击区域
                if idx not in self.marked_points:
                    if len(self.marked_points) < 3:
                        self.marked_points.append(idx)  # 保持点击顺序
                        print(f"添加点 {idx}, 当前点序列: {self.marked_points}")
                        self.update_canvas()
                    else:
                        QMessageBox.warning(self, "提示", "已选择3个点，请先添加到表格或清除选择")
                return

    def update_canvas(self):
        """更新画布显示"""
        if self.canvas is None:
            return
            
        canvas = self.canvas.copy()
        color = ( 0, 0,255)  # 蓝色 (BGR格式)
        
        # 绘制已标记的点和序号
        for i, idx in enumerate(self.marked_points):
            x, y = self.coordinates[idx]
            x += 10
            # 绘制蓝色圆点
            cv2.circle(canvas, (int(x), int(y)), 5, color, -1, lineType=cv2.LINE_AA)
            # 绘制序号
            cv2.putText(canvas, str(i+1), (int(x-10), int(y-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        height, width, channel = canvas.shape
        bytes_per_line = channel * width
        q_image = QImage(canvas.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.ui.posture.setPixmap(QPixmap.fromImage(q_image))

    def on_add_angle_clicked(self):
        """添加角度到表格"""
        if len(self.marked_points) != 3:
            QMessageBox.warning(self, "提示", "请先选择3个点！")
            return
        
        if self.add_button_count >= 5:
            QMessageBox.warning(self, "提示", "最多只能添加5个角度")
            return
            
        self.add_button_count += 1
        # 使用点击顺序创建角度字符串
        angle_str = f"∠{self.marked_points[0]}-{self.marked_points[1]}-{self.marked_points[2]}"
        
        row_count = self.table_model.rowCount()
        self.table_model.insertRow(row_count)
        self.table_model.setItem(row_count, 0, QStandardItem(angle_str))
        self.table_model.setItem(row_count, 1, QStandardItem("0.0"))
        self.table_model.setItem(row_count, 2, QStandardItem("0"))
        
        # 清除已标记的点
        self.marked_points = []
        self.update_canvas()

    def on_delete_clicked(self):
        """删除选中的表格��"""
        indexes = self.ui.tableView.selectedIndexes()
        if not indexes:
            QMessageBox.warning(self, "提示", "请先选择要删除的行")
            return
            
        row = indexes[0].row()
        self.table_model.removeRow(row)
        self.add_button_count -= 1
        
        # 清除已标记的点
        self.marked_points = []
        self.update_canvas()

    def clear_selection(self):
        """清除当前选择的点"""
        self.marked_points = []
        self.update_canvas()

    def on_save_clicked(self):
        """保存数据到JSON文件"""
        model = self.ui.tableView.model()
        if model is None or model.rowCount() == 0:
            QMessageBox.warning(self, "提示", "表格中没有数据可保存！")
            return

        json_data = []
        for row in range(model.rowCount()):
            key_angle_item = model.item(row, 0)
            weight_item = model.item(row, 1)
            score_item = model.item(row, 2)

            key_angle = key_angle_item.text() if key_angle_item else ""
            weight = float(weight_item.text()) if weight_item else 0.0
            score = int(score_item.text()) if score_item else 0

            # 直接使用角度字符串中的点序列
            if key_angle.startswith("∠"):
                angle_points = [int(p) for p in key_angle[1:].split("-")]
            else:
                angle_points = []

            json_data.append({
                "key_angle": angle_points,
                "weight": weight,
                "angle": score
            })

        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存权重文件", "", "JSON Files (*.json)")
        if not save_path:
            return

        try:
            with open(save_path, "w", encoding="utf-8") as file:
                json.dump(json_data, file, indent=4, ensure_ascii=False)
            QMessageBox.information(self, "成功", f"权重文件已保存到: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存文件失败: {e}")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = WeightApp()
    main_window.show()
    sys.exit(app.exec())
