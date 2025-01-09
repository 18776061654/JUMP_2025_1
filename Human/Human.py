# Copyright (c) OpenMMLab. All rights reserved.
import torch
import cv2
import argparse
from configparser import ConfigParser
args = argparse.Namespace()
import numpy as np
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from configparser import ConfigParser

KEYPOINT_CONNECTIONS = [
    (0, 1),  # 鼻子到左眼
    (0, 2),  # 鼻子到右眼
    (1, 2),  # 左眼到右眼
    (1, 3),  # 左眼到左耳
    (2, 4),  # 右眼到右耳
    (3, 5),  # 左耳到左肩
    (4, 6),  # 右耳到右肩
    (5, 6),  # 左肩到右肩
    (5, 7),  # 左肩到左肘
    (7, 9),  # 左肘到左腕
    (6, 8),  # 右肩到右肘
    (8, 10), # 右肘到右腕
    (5, 11), # 左肩到左髋
    (6, 12), # 右肩到右髋
    (11, 12),# 左髋到右髋
    (11, 13),# 左髋到左膝
    (13, 15),# 左膝到左踝
    (12, 14),# 右髋到右膝
    (14, 16),# 右膝到右踝
]



class Human(object):
    def __init__(self,config_path):
        """初始化，加载配置"""
        config = ConfigParser()
        config.read(config_path)
        
        human_config = config['Human']

        # human_config初始化
        self.det_config = human_config['det_config']
        self.det_model  = human_config['det_model']
        self.pose_config = human_config['pose_config']
        self.pose_model = human_config['pose_model']
        self.det_cat_id = int(human_config['det_cat_id'])
        self.bbox_thr = float(human_config['bbox_thr'])
        self.nms_thr = float(human_config['nms_thr'])        
        self.kpt_thr = float(human_config['kpt_thr'])
        self.draw_heatmap = bool(human_config['draw_heatmap']) #控制模型是否输出热图信息
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # 打印设备信息
        print(f"Using device: {self.device}")


    def init_model(self):
        """初始化模型，包括目标检测和人体姿态识别"""
        self.detector = init_detector(
            self.det_config, self.det_model, device=self.device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        self.pose_estimator = init_pose_estimator(
            self.pose_config,
            self.pose_model,
            device=self.device,
            cfg_options=dict(
                model=dict(test_cfg=dict(output_heatmaps=self.draw_heatmap))))
        


    def get_bbox_one_image(self, img):
            """
            获取单张图片中的目标检测框

            :param img: 输入的图像
            :return: 目标检测框数组，格式为[[x1, y1, x2, y2], ...]
            """
            # 使用检测模型进行推理
            det_result = inference_detector(self.detector, img)
            # 将推理结果转换为numpy数组
            pred_instance = det_result.pred_instances.cpu().numpy()
            # 合并检测框和分数
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            # 过滤出符合类别和分数阈值的检测框
            bboxes = bboxes[np.logical_and(pred_instance.labels == self.det_cat_id,
                                        pred_instance.scores > self.bbox_thr)]
            # 使用非极大值抑制去除冗余检测框
            bboxes = bboxes[nms(bboxes, self.nms_thr), :4]
            
            return bboxes 
        
    def draw_box(self, img, bboxes, box_color=(0, 255, 0), box_thickness=4):
        """
        在图像上绘制目标检测框

        :param img: 输入的图像
        :param bboxes: 目标检测框，格式为[[x1, y1, x2, y2], ...]
        :param box_color: 矩形框的颜色
        :param box_thickness: 矩形框的线条粗细
        :return: 绘制了目标检测框的图像
        """
        # 复制图像，避免修改原始图像
        img_with_boxes = img.copy()

        # 检查 bboxes 是否为空
        if bboxes is None or len(bboxes) == 0:
            print("No bounding boxes to draw.")
            return img_with_boxes

        # 绘制每个目标检测框
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.astype(int)
            # 检查坐标是否在图像范围内
            if x1 < 0 or y1 < 0 or x2 > img_with_boxes.shape[1] or y2 > img_with_boxes.shape[0]:
                print(f"Bounding box {bbox} is out of image bounds.")
                continue
            # 在图像上绘制矩形框
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color=box_color, thickness=box_thickness)
        
        return img_with_boxes



    def get_pose_one_image(self, img, bboxes=None):
        """
        获取单张图片中的人体姿态

        :param img: 输入的图像
        :param bboxes: 目标检测框，格式为[[x1, y1, x2, y2], ...],默认为None
        :return: 包含预测关节点的实例数据
        """
        # 预测关键点
        pose_results = inference_topdown(self.pose_estimator, img, bboxes)
        data_samples = merge_data_samples(pose_results)
        
        pred_instances = data_samples.get('pred_instances')
        pose_instances = split_instances(pred_instances)

        # 如果没有检测到实例，返回None
        return pose_instances
    


    def draw_pose(self, img, pose_instances, point_color=(0, 0, 255, 255), point_size=4, line_color=(0, 255, 0, 255), line_thickness=2):
        """
        在图像上绘制人体姿态关键点和连线，并整体向右移动10个像素。

        :param img: 输入的图像
        :param pose_instances: 包含预测关节点的实例数据
        :param point_color: 关键点的颜色 (RGBA)
        :param line_color: 连线的颜色 (RGBA)
        :return: 绘制了关键点和连线的图像
        """

        # 检查图像是否为三通道，如果是，则转换为四通道
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        overlay = img.copy()  # 创建一个覆盖层用于绘制带透明度的关键点和连线

        for instance in pose_instances:
            keypoints = instance['keypoints']
            keypoint_scores = instance['keypoint_scores']

            # 绘制连线
            for (start_idx, end_idx) in KEYPOINT_CONNECTIONS:
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                start_score = keypoint_scores[start_idx]
                end_score = keypoint_scores[end_idx]
                if start_score > self.kpt_thr and end_score > self.kpt_thr:  # 只绘制置信度大于kpt_thr的连线
                    start_x, start_y = int(start_point[0]) + 10, int(start_point[1])
                    end_x, end_y = int(end_point[0]) + 10, int(end_point[1])
                    cv2.line(img, (start_x, start_y), (end_x, end_y), line_color[:3], line_thickness, lineType=cv2.LINE_AA)

            # 绘制关键点
            for keypoint, score in zip(keypoints, keypoint_scores):
                x, y = keypoint
                if score > self.kpt_thr:  # 只绘制置信度大于kpt_thr的关键点
                    x += 10  # 整体向右移动10个像素
                    cv2.circle(img, (int(x), int(y)), point_size, point_color[:3], -1, lineType=cv2.LINE_AA)
                    
                    # 手动设置透明度
                    if img.shape[2] == 4:  # 确保图像有透明度通道
                        alpha_channel = img[max(0, int(y) - point_size):int(y) + point_size, max(0, int(x) - point_size):int(x) + point_size, 3]
                        alpha_channel[:] = point_color[3]

        return img

    
    def calculate_angle(self, keypoint1, keypoint2, keypoint3):
        """
        计算由三个关键点形成的角度。keypoint2 是中间点。
        
        :param keypoint1: 第一个关键点的坐标 (x1, y1)
        :param keypoint2: 第二个关键点的坐标 (x2, y2)，也是中间点
        :param keypoint3: 第三个关键点的坐标 (x3, y3)
        :return: 中间点的角度（以度为单位）
        """
        x1, y1 = keypoint1
        x2, y2 = keypoint2
        x3, y3 = keypoint3
        
        # 向量 u: keypoint2 -> keypoint1
        u = np.array([x1 - x2, y1 - y2])
        # 向量 v: keypoint2 -> keypoint3
        v = np.array([x3 - x2, y3 - y2])
        
        # 计算向量的夹角
        cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        angle = np.arccos(cos_theta)
        
        # 将弧度转换为度
        angle_degrees = np.degrees(angle)
        
        return angle_degrees
    
    def get_head_keypoints(self, keypoints):
        """
        获取头部关键点并计算质心
        
        :param keypoints: 包含17个点坐标的列表,每个点是一个(x, y)元组
        :return: 头部关键点的质心 (cx, cy)
        """
        # 选择头部关键点（假设是第0到4个点）
        head_keypoints = keypoints[:5]
        
        # 计算质心
        x_coords = [kp[0] for kp in head_keypoints if kp is not None]
        y_coords = [kp[1] for kp in head_keypoints if kp is not None]

        if len(x_coords) == 0 or len(y_coords) == 0:
            raise ValueError("没有有效的头部关键点坐标")

        cx = sum(x_coords) / len(x_coords)
        cy = sum(y_coords) / len(y_coords)

        return cx, cy
    




