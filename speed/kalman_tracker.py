import numpy as np
import cv2

class KalmanTracker:
    
    def __init__(self):
        """初始化卡尔曼滤波器"""
        self.kalman = cv2.KalmanFilter(4, 2)  # 状态维度4，观测维度2
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        self.prediction = None  # 预测值
        self.corrected = None  # 校正后的值

    def update(self, keypoint):
        """使用观测值更新卡尔曼滤波器"""
        measurement = np.array([[np.float32(keypoint[0])],
                                [np.float32(keypoint[1])]], np.float32)
        
        if self.prediction is None:
            # 初始化预测值
            self.kalman.statePre = np.array([[np.float32(keypoint[0])],
                                             [np.float32(keypoint[1])],
                                             [0],
                                             [0]], np.float32)
            self.corrected = self.kalman.correct(measurement)
            return keypoint
        else:
            # 更新卡尔曼滤波器，校正值
            self.prediction = self.kalman.predict()
            self.corrected = self.kalman.correct(measurement)
            corrected_keypoint = (self.corrected[0, 0], self.corrected[1, 0])
            return corrected_keypoint

    def predict(self):
        """返回卡尔曼滤波器的预测结果"""
        return self.prediction
