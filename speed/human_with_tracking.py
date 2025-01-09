from configparser import ConfigParser
import numpy as np
from speed.kalman_tracker import KalmanTracker
from Human.Human import Human

class HumanWithTracking(Human):
    def __init__(self, config_path):
        super().__init__(config_path)
        
        config = ConfigParser()
        config.read(config_path)

        # 初始化卡尔曼滤波器追踪器
        self.tracker = KalmanTracker()
        self.previous_keypoint = None

        pixel_point1 = eval(config.get('Runway', 'pixel_point1_1'))
        pixel_point2 = eval(config.get('Runway', 'pixel_point1_2'))
        real_distance = config.getfloat('Runway', 'real_distance_1')

        self.conversion_ratio = self.calculate_conversion_ratio(pixel_point1, pixel_point2, real_distance)

    def calculate_conversion_ratio(self, pixel_point1, pixel_point2, real_distance):
        pixel_distance = np.sqrt((pixel_point2[0] - pixel_point1[0]) ** 2 +
                                 (pixel_point2[1] - pixel_point1[1]) ** 2)
        conversion_ratio = real_distance / pixel_distance
        return conversion_ratio

    def process_frame(self, img, fps):
        bboxes = self.get_bbox_one_image(img)
        if len(bboxes) == 0:
            return None

        pose_instances = self.get_pose_one_image(img, bboxes)
        if len(pose_instances) == 0:
            return None

        current_keypoint = pose_instances[0]['keypoints'][0]
        if current_keypoint is None:
            return None

        corrected_keypoint = self.tracker.update(current_keypoint)

        if self.previous_keypoint is not None:
            speed = self.calculate_speed(corrected_keypoint, self.previous_keypoint, fps)
        else:
            speed = 0  # 第一帧速度为0

        self.previous_keypoint = corrected_keypoint
        return speed

    def calculate_speed(self, current_keypoint, previous_keypoint, fps):
        pixel_distance = np.sqrt((current_keypoint[0] - previous_keypoint[0]) ** 2 +
                                 (current_keypoint[1] - previous_keypoint[1]) ** 2)
        real_distance = pixel_distance * self.conversion_ratio
        time_interval = 1.0 / fps
        speed = real_distance / time_interval
        return speed

    def process_batch(self, frames, fps):
        """批量处理帧并计算速度，保持前后帧的连续性"""
        speeds = []
        for i in range(1, len(frames)):
            current_frame = frames[i]
            previous_frame = frames[i - 1]

            # 处理上一帧以获取关键点
            self.process_frame(previous_frame, fps)
            
            # 处理当前帧以计算速度
            speed = self.process_frame(current_frame, fps)

            if speed is not None:
                speeds.append(speed)
        
        return speeds
