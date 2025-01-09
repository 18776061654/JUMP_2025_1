import cv2
import numpy as np
from Human.Human import Human

class HumanJumpAnalyzer():
    def __init__(self):
        self.human = Human(config_path='config/config.ini')
        self.human.init_model()
        self.last_bbox = None  # 用于记录上一帧的检测框，确保连续检测到同一个目标
        self.is_tracking = False  # 是否正在跟踪跳远者

    def process_jump_video(self, video_path, output_image_path):
        """
        处理跳远视频并实时显示跳远者的框中心
        :param video_path: 输入视频路径
        :param output_image_path: 输出绘制了轨迹的图片路径
        """
        if not video_path:
            return None, "视频路径无效"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "无法打开视频文件"

        # 获取视频帧宽高
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        bbox_positions = []  # 用于存储检测框中心的坐标轨迹
        frame_count = 0  # 处理帧数的计数

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"正在处理第 {frame_count} 帧")

            # 获取目标检测框
            bboxes = self.human.get_bbox_one_image(frame)
            print(f"第 {frame_count} 帧检测到的目标数量: {len(bboxes)}")

            if len(bboxes) == 0:
                continue

            # 选择最大的检测框作为跳远者
            largest_bbox = self.select_largest_bbox(bboxes)
            if largest_bbox is None:
                continue

            # 判断当前框是否是跳远者的框
            if self.is_tracking_larger_object(largest_bbox):
                center_x, center_y = self.get_bbox_center(largest_bbox)
                center_x, center_y = int(center_x), int(center_y)
                bbox_positions.append((center_x, center_y))
                print(f"记录的跳远者框中心点: ({center_x}, {center_y})")

                # 在当前帧上绘制中心点
                cv2.circle(frame, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=-1)
                cv2.putText(frame, f"Center: ({center_x}, {center_y})", (center_x - 50, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 显示处理后的帧，包含检测的中心点
            cv2.imshow("Jump Analysis", frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 更新最后检测到的跳远者框
            self.last_bbox = largest_bbox

        cap.release()
        cv2.destroyAllWindows()

        if len(bbox_positions) == 0:
            return None, "视频中未检测到任何目标"

        # 基于运动轨迹过滤目标
        if not self.filter_moving_objects(bbox_positions):
            return None, "未检测到跳远者的运动"

        # 创建一个新的白色面板，假设高度为 500 像素，宽度为 1000 像素
        panel_height = 500
        panel_width = 1000
        white_panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255  # 白色背景

        # 在面板上绘制坐标轴
        self.draw_axes(white_panel, panel_width, panel_height)

        # 将轨迹映射到白色面板上并绘制
        self.draw_trajectory_on_panel(bbox_positions, white_panel, frame_width, frame_height)

        # 保存带有轨迹和坐标轴的白色面板为图片
        cv2.imwrite(output_image_path, white_panel)

        return "轨迹绘制完成"

    def select_largest_bbox(self, bboxes):
        """
        选择最大的检测框作为跳远者
        :param bboxes: 检测框列表 [(x_min, y_min, x_max, y_max), ...]
        :return: 最大的检测框
        """
        if len(bboxes) == 0:
            return None
        # 按照检测框的面积（宽度 * 高度）选择最大的框
        largest_bbox = max(bboxes, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        return largest_bbox

    def is_tracking_larger_object(self, current_bbox, size_threshold=0.8):
        """
        判断是否应该跟踪当前检测框（跳远者），忽略较小的框
        :param current_bbox: 当前帧中的检测框
        :param size_threshold: 大小阈值，用于确定跳远者框
        :return: 是否应该跟踪
        """
        if self.last_bbox is None:
            self.is_tracking = True
            return True  # 初始时跟踪最大的框

        # 计算当前框和上一个框的面积
        current_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
        last_area = (self.last_bbox[2] - self.last_bbox[0]) * (self.last_bbox[3] - self.last_bbox[1])

        # 如果当前框大于阈值，或当前框面积显著大于前一帧的检测框，继续跟踪
        if current_area > size_threshold * last_area:
            self.is_tracking = True
            return True
        else:
            # 如果跳远者离开视野，不再跟踪
            self.is_tracking = False
            return False

    def get_bbox_center(self, bbox):
        """
        获取检测框的中心坐标
        :param bbox: (x_min, y_min, x_max, y_max)
        :return: (center_x, center_y)
        """
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        return center_x, center_y

    def filter_moving_objects(self, bbox_positions, min_movement_threshold=20):
        """
        过滤掉没有明显移动的目标，只保留有明显运动的目标（跳远者）
        :param bbox_positions: 检测框的质心坐标轨迹列表 [(x1, y1), (x2, y2), ...]
        :param min_movement_threshold: 移动距离的最小阈值
        :return: 是否有明显运动
        """
        if len(bbox_positions) < 2:
            return False
        
        total_movement = 0
        for i in range(1, len(bbox_positions)):
            x1, y1 = bbox_positions[i-1]
            x2, y2 = bbox_positions[i]
            # 计算连续帧之间的移动距离
            movement = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_movement += movement

        # 如果运动距离超过阈值，则认为是跳远者
        return total_movement > min_movement_threshold

    def draw_axes(self, panel, panel_width, panel_height):
        """
        在白色面板上绘制坐标轴，并添加刻度和标签
        :param panel: 白色面板图像
        :param panel_width: 面板宽度
        :param panel_height: 面板高度
        """
        # X轴（距离）和 Y轴（高度）的位置
        origin = (50, panel_height - 50)  # 原点位置
        axis_length = panel_width - 100  # X轴的长度
        axis_height = panel_height - 100  # Y轴的长度

        # 绘制X轴和Y轴
        cv2.line(panel, origin, (origin[0] + axis_length, origin[1]), (0, 0, 0), 2)  # X轴
        cv2.line(panel, origin, (origin[0], origin[1] - axis_height), (0, 0, 0), 2)  # Y轴

        # 绘制X轴刻度和标签
        num_x_ticks = 10  # X轴刻度数量
        x_step = axis_length // num_x_ticks
        for i in range(1, num_x_ticks + 1):
            x_pos = origin[0] + i * x_step
            cv2.line(panel, (x_pos, origin[1]), (x_pos, origin[1] + 10), (0, 0, 0), 2)  # 刻度线
            cv2.putText(panel, f"{i * 10}", (x_pos - 10, origin[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 绘制Y轴刻度和标签
        num_y_ticks = 10  # Y轴刻度数量
        y_step = axis_height // num_y_ticks
        for i in range(1, num_y_ticks + 1):
            y_pos = origin[1] - i * y_step
            cv2.line(panel, (origin[0], y_pos), (origin[0] - 10, y_pos), (0, 0, 0), 2)  # 刻度线
            cv2.putText(panel, f"{i * 10}", (origin[0] - 40, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 添加轴标签
        cv2.putText(panel, "Distance", (panel_width - 120, panel_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(panel, "Height", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def draw_trajectory_on_panel(self, bbox_positions, panel, frame_width, frame_height):
        """
        将轨迹绘制到白色面板上，X轴为距离，Y轴为高度
        :param bbox_positions: 目标检测框的质心坐标
        :param panel: 白色面板图像
        :param frame_width: 视频帧的宽度
        :param frame_height: 视频帧的高度
        """
        panel_height, panel_width, _ = panel.shape
        origin = (50, panel_height - 50)  # 原点位置
        axis_length = panel_width - 100  # X轴的长度
        axis_height = panel_height - 100  # Y轴的长度

        # 将视频中的坐标映射到白色面板上
        for i in range(1, len(bbox_positions)):
            x1, y1 = bbox_positions[i-1]
            x2, y2 = bbox_positions[i]

            # 将视频中的坐标比例映射到白色面板上
            mapped_x1 = int((x1 / frame_width) * axis_length + origin[0])
            mapped_y1 = int(origin[1] - (frame_height - y1) / frame_height * axis_height)  # Y轴映射反转
            mapped_x2 = int((x2 / frame_width) * axis_length + origin[0])
            mapped_y2 = int(origin[1] - (frame_height - y2) / frame_height * axis_height)  # Y轴映射反转

            # 在面板上绘制轨迹线
            cv2.line(panel, (mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (0, 0, 255), 2)


# 使用示例
if __name__ == "__main__":
    video_path = 'd:/Program Files (x86)/Study/Jump-video/jump-qiu/athletes3.mp4'
    output_image_path = 'trajectory_with_axes.png'  # 输出图片路径
    
    human_analyzer = HumanJumpAnalyzer()
    result = human_analyzer.process_jump_video(video_path, output_image_path)
    
    print(result)
