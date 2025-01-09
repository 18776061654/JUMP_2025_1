import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox,
                              QLineEdit, QGridLayout, QFrame)
from PySide6.QtGui import QPainter, QColor, QPen, QPixmap
from PySide6.QtCore import Qt, QRect, QPoint, Signal
import cv2
import os
import json
from datetime import datetime

class DraggableLine:
    def __init__(self, x, color, is_start=True, stage_index=0):
        self.x = x
        self.color = color
        self.is_selected = False
        self.handle_radius = 8  # 增大控制点半径，更容易点击
        self.is_start = is_start
        self.stage_index = stage_index
        
    def contains(self, x, y, height):
        """检查点击位置是否在控制点范围内"""
        # 根据阶段索引计算垂直偏移
        vertical_offset = 30 * self.stage_index
        center_y = height // 2 + vertical_offset
        
        # 放宽点击检测范围
        dx = abs(x - self.x)
        dy = abs(y - center_y)
        
        # 打印调试信息
        print(f"点击位置: ({x}, {y})")
        print(f"控制点位置: ({self.x}, {center_y})")
        print(f"距离: dx={dx}, dy={dy}")
        
        # 增大检测范围
        return dx <= self.handle_radius * 3 and dy <= self.handle_radius * 3

    def get_handle_position(self, height):
        vertical_offset = 30 * self.stage_index
        return QPoint(self.x, height // 2 + vertical_offset)

class StageSelector(QMainWindow):
    MAX_STAGES = 5  # 定义最大区间数量
    
    def __init__(self, image_path=None):
        super().__init__()
        self.setWindowTitle("跳远阶段划分")
        self.resize(1200, 800)

        # 如果没有提供图片路径，打开文件选择对话框
        if image_path is None or not os.path.exists(image_path):
            image_path = self.get_image_path()
            if not image_path:  # 如果用户取消选择
                raise ValueError("未选择图片文件")

        # 加载图片
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"无法加载图片: {image_path}")
            
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        height, width, channel = self.image.shape
        self.aspect_ratio = width / height
        
        # 创建主窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建布局
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # 创建控制板
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.main_layout.addWidget(self.control_panel, stretch=1)
        
        # 添加控制按钮
        self.add_button = QPushButton("添加区间")
        self.remove_button = QPushButton("删除区间")
        self.save_button = QPushButton("保存区间")
        
        self.control_layout.addWidget(self.add_button)
        self.control_layout.addWidget(self.remove_button)
        
        # 添加区间编辑区域
        self.stage_edit_widget = QWidget()
        self.stage_edit_layout = QGridLayout(self.stage_edit_widget)
        self.stage_edit_layout.setSpacing(10)
        
        # 添加标题
        self.stage_edit_layout.addWidget(QLabel("区间ID"), 0, 0)
        self.stage_edit_layout.addWidget(QLabel("名称"), 0, 1)
        self.stage_edit_layout.addWidget(QLabel("范围"), 0, 2)
        
        # 存储区间编辑控件
        self.stage_editors = []
        
        # 初始化阶段
        self.stages = ["takeoff", "flight", "landing"]  # 默认英文ID
        self.stage_names = ["起跳", "空中", "落地"]  # 对应的中文名称
        
        # 连接信号
        self.add_button.clicked.connect(self.add_stage)
        self.remove_button.clicked.connect(self.remove_stage)
        self.save_button.clicked.connect(self.save_stages)
        
        # 更新界面
        self.control_layout.addWidget(self.stage_edit_widget)
        self.control_layout.addWidget(self.save_button)
        self.control_layout.addStretch()
        
        # 初始化图像显示区域
        self.image_widget = ImageWidget(self.image, self)
        self.main_layout.addWidget(self.image_widget, stretch=4)
        
        # 连接区间变化信号
        self.image_widget.ranges_changed.connect(self.update_range_labels)
        
        # 初始化显示
        self.image_widget.update_stages(self.stages)
        self.update_stage_editors()
        self.update_button_states()

    def get_image_path(self):
        """打开文件选择对话框让用户选择图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*.*)"
        )
        return file_path

    def update_stage_editors(self):
        """更新区间编辑控件"""
        # 清��现有的编辑控件
        for editor in self.stage_editors:
            for widget in editor:
                widget.setParent(None)
        self.stage_editors.clear()
        
        # 为每个阶段创建编辑控件
        for i, (stage_id, stage_name) in enumerate(zip(self.stages, self.stage_names)):
            # ID标签
            id_label = QLabel(stage_id)
            
            # 名称编辑框
            name_edit = QLineEdit(stage_name)
            name_edit.textChanged.connect(lambda text, idx=i: self.update_stage_name(idx, text))
            
            # 范围标签
            range_info = self.get_stage_range_info(i)
            range_label = QLabel(range_info)
            
            # 添加到布局
            self.stage_edit_layout.addWidget(id_label, i + 1, 0)
            self.stage_edit_layout.addWidget(name_edit, i + 1, 1)
            self.stage_edit_layout.addWidget(range_label, i + 1, 2)
            
            # 保存编辑控件引用
            self.stage_editors.append((id_label, name_edit, range_label))

    def add_stage_after(self, index):
        """在指定位置后添加新阶段"""
        if len(self.stages) < self.MAX_STAGES:
            new_id = f"stage_{len(self.stages) + 1}"
            new_name = f"阶段{len(self.stages) + 1}"
            
            self.stages.insert(index + 1, new_id)
            self.stage_names.insert(index + 1, new_name)
            
            self.image_widget.update_stages(self.stages)
            self.update_stage_editors()

    def remove_stage_at(self, index):
        """删除指定位置的阶段"""
        if len(self.stages) > 2:
            self.stages.pop(index)
            self.stage_names.pop(index)
            
            self.image_widget.update_stages(self.stages)
            self.update_stage_editors()

    def update_stage_name(self, index, new_name):
        """更新阶段名称"""
        if 0 <= index < len(self.stage_names):
            self.stage_names[index] = new_name
            self.update_range_labels()

    def get_stage_range_info(self, index):
        """获取阶段范围信息"""
        stage_info = self.image_widget.get_stage_info()
        if self.stages[index] in stage_info:
            info = stage_info[self.stages[index]]
            return f"{info['start']:.2f} - {info['end']:.2f}"
        return "N/A"

    def update_range_labels(self):
        """更新所有范围标签"""
        stage_info = self.image_widget.get_stage_info()
        for i, (_, _, range_label) in enumerate(self.stage_editors):
            if self.stages[i] in stage_info:
                info = stage_info[self.stages[i]]
                range_label.setText(f"{info['start']:.2f} - {info['end']:.2f}")
            else:
                range_label.setText("N/A")

    def save_stages(self):
        """保存阶段划分到JSON文件"""
        try:
            stage_info = self.image_widget.get_stage_info()
            
            # 准备保存的数据
            save_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_stages': len(self.stages),
                'stages': [{
                    'id': stage_id,
                    'name': stage_name,
                    'info': stage_info[stage_id]
                } for stage_id, stage_name in zip(self.stages, self.stage_names)],
                'image_size': {
                    'width': self.image_widget.pixmap.width(),
                    'height': self.image_widget.pixmap.height()
                }
            }
            
            # 打开保存文件对话框
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存阶段划分",
                "",
                "JSON文件 (*.json);;所有文件 (*.*)"
            )
            
            if file_path:
                # 确保文件扩展名为.json
                if not file_path.endswith('.json'):
                    file_path += '.json'
                
                # 确保目录存在
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # 保存到JSON文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=4)
                
                print(f"阶段划分已保存到: {file_path}")
                
                # 显示成功消息
                QMessageBox.information(
                    self,
                    "保存成功",
                    f"阶段划分已成功保存到:\n{file_path}"
                )
                
        except Exception as e:
            # 显示错误消息
            QMessageBox.critical(
                self,
                "保存失败",
                f"保存阶段划分时出错:\n{str(e)}"
            )
            print(f"保存阶段划分时出错: {str(e)}")

    def add_stage(self):
        """添加新的阶段"""
        if len(self.stages) < self.MAX_STAGES:
            new_id = f"stage_{len(self.stages) + 1}"
            new_name = f"阶段{len(self.stages) + 1}"
            self.stages.append(new_id)
            self.stage_names.append(new_name)
            self.image_widget.update_stages(self.stages)
            self.update_stage_editors()
            self.update_button_states()

    def remove_stage(self):
        """删除最后一个阶段"""
        if len(self.stages) > 2:
            self.stages.pop()
            self.stage_names.pop()
            self.image_widget.update_stages(self.stages)
            self.update_stage_editors()
            self.update_button_states()

    def update_button_states(self):
        """更新按钮状态"""
        self.add_button.setEnabled(len(self.stages) < self.MAX_STAGES)
        self.remove_button.setEnabled(len(self.stages) > 2)

class ImageWidget(QWidget):
    # 添加信号用于通知区间变化
    ranges_changed = Signal()
    
    # 添加最小区间宽度常量
    MIN_STAGE_WIDTH = 50  # 最小区间宽度（像素）
    
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image
        self.pixmap = QPixmap.fromImage(self.convert_cv_qt(image))
        self.stage_lines = []  # 每个阶段的起始和结束线
        self.selected_line = None
        self.stages = []
        self.stage_colors = [
            QColor(255, 0, 0, 50),
            QColor(0, 255, 0, 50),
            QColor(0, 0, 255, 50),
            QColor(255, 255, 0, 50),
            QColor(255, 0, 255, 50),
        ]
        
        self.setMouseTracking(True)

    def convert_cv_qt(self, cv_img):
        """将OpenCV图像换为Qt图像"""
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        from PySide6.QtGui import QImage
        return QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def update_stages(self, stages):
        """更新阶段划分"""
        self.stages = stages
        self.stage_lines = []
        
        # 获取图片的实际显示区域
        scaled_pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio)
        x_offset = (self.width() - scaled_pixmap.width()) // 2
        available_width = scaled_pixmap.width()
        
        # 计算每个阶段的宽度（均匀分布）
        segment_width = available_width / len(stages)
        
        # 为每个阶段创建两条线（起始和结束）
        for i, stage in enumerate(stages):
            # 计算每个阶段的起始和结束位置
            start_x = int(x_offset + (segment_width * i))  # 确保是整数
            end_x = int(x_offset + (segment_width * (i + 1)))  # 确保是整数
            
            # 创建一对线（起始和结束）
            start_line = DraggableLine(start_x, self.stage_colors[i], True, i)
            end_line = DraggableLine(end_x, self.stage_colors[i], False, i)
            self.stage_lines.append((start_line, end_line))
        
        # 强制更新显示
        self.update()
        # 发出信号通知范围已更新
        self.ranges_changed.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # 添加抗锯齿
        
        # 绘制图片
        scaled_pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio)
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)
        
        # 绘制区间
        if self.stage_lines:
            rect_height = scaled_pixmap.height()
            rect_y = y
            
            # 绘制每个阶段的区域和控制点
            for i, (start_line, end_line) in enumerate(self.stage_lines):
                # 绘制区域
                painter.fillRect(start_line.x, rect_y,
                               end_line.x - start_line.x,
                               rect_height,
                               self.stage_colors[i])
                
                # 绘制控制线
                pen = QPen(Qt.black, 2)
                painter.setPen(pen)
                
                # 绘制起始和结束线
                for line in [start_line, end_line]:
                    # 绘制垂直线
                    painter.drawLine(line.x, rect_y, line.x, rect_y + rect_height)
                    
                    # 绘制控制点
                    handle_pos = line.get_handle_position(rect_height)
                    if line.is_selected:
                        painter.setBrush(Qt.red)
                    else:
                        painter.setBrush(Qt.white)
                    painter.drawEllipse(handle_pos, line.handle_radius, line.handle_radius)
                    
                    # 绘制标签
                    label = "S" if line.is_start else "E"
                    painter.drawText(
                        handle_pos.x() - 10,
                        handle_pos.y() - 10,
                        label
                    )

    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        # 获取点击位置
        pos = event.position()
        x = int(pos.x())
        
        # 获取图片的实际显示区域
        scaled_pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio)
        x_offset = (self.width() - scaled_pixmap.width()) // 2
        
        # 检查是否点击在分隔线附近（使用更大的检测范围）
        click_range = 10  # 增大点击检测范围
        
        for start_line, end_line in self.stage_lines:
            # 检查是否点击在任一分隔线附近
            if abs(x - start_line.x) <= click_range:
                self.selected_line = start_line
                start_line.is_selected = True
                self.update()
                break
            elif abs(x - end_line.x) <= click_range:
                self.selected_line = end_line
                end_line.is_selected = True
                self.update()
                break

    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        if self.selected_line:
            pos = event.position()
            x = int(pos.x())
            
            scaled_pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio)
            x_offset = (self.width() - scaled_pixmap.width()) // 2
            
            # 限制在图片范围内
            min_x = x_offset
            max_x = x_offset + scaled_pixmap.width()
            
            # 确保新位置在有效范围内
            new_x = max(min_x, min(x, max_x))
            
            # 更新位置并触发重绘
            if self.selected_line.x != new_x:
                self.selected_line.x = new_x
                self.update()
                self.ranges_changed.emit()

    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if self.selected_line:
            self.selected_line.is_selected = False
            self.selected_line = None
            self.update()

    def get_stage_info(self):
        """获取阶段划分信息"""
        stage_info = {}
        scaled_pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio)
        x_offset = (self.width() - scaled_pixmap.width()) // 2
        image_width = scaled_pixmap.width()
        
        for i, (stage_id, (start_line, end_line)) in enumerate(zip(self.stages, self.stage_lines)):
            # 计算相对位置
            start_pos = (start_line.x - x_offset) / image_width
            end_pos = (end_line.x - x_offset) / image_width
            
            stage_info[stage_id] = {
                'name': stage_id,
                'start': start_pos,
                'end': end_pos,
                'pixel_start': start_line.x - x_offset,
                'pixel_end': end_line.x - x_offset
            }
        
        return stage_info

def main():
    app = QApplication(sys.argv)
    try:
        window = StageSelector()  # 不传入图片路径，让用户选择
        window.show()
        sys.exit(app.exec())
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 