import cv2
import numpy as np
from typing import List, Tuple, Dict
import json
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import tkinter as tk
from tkinter import filedialog
import os
from Human.Human import Human
from posture.JumperTracker import Sort, KalmanBoxTracker
from posture.BiLSTMActionSegmentation import BiLSTMActionSegmentation
from posture.DTWCosineMatcher import DTWCosineMatcher
from posture.RelativeL2Scorer import RelativeL2Scorer
import torch
import datetime

class JumpAnalyzer:
    def __init__(self):
        # 初始化Human类
        self.human = Human(config_path='config/config.ini')
        self.human.init_model()
        
        # 定义关键点连接关系
        self.KEYPOINT_CONNECTIONS = [
            (5, 6),   # 左肩到右肩
            (5, 7),   # 左肩到左肘
            (7, 9),   # 左肘到左腕
            (6, 8),   # 右肩到右肘
            (8, 10),  # 右肘到右腕
            (5, 11),  # 左肩到左
            (6, 12),  # 右肩到右髋
            (11, 12), # 左髋到右髋
            (11, 13), # 左髋到左膝
            (13, 15), # 左膝到左踝
            (12, 14), # 右髋到右膝
            (14, 16)  # 右膝到右踝
        ]
        
        # 起跳线预备框的位置
        self.start_zone = None
        # 追踪状态
        self.tracking_active = False
        # 当前追踪的目标ID
        self.target_id = None
        # 追踪器
        self.tracker = Sort()
        # 记录轨迹和姿态数据
        self.trajectory = []
        
        # 关键点索引映射
        self.key_point_indices = {
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16
        }
        
        self.total_frames = 0
        self.current_frame = 0
        
        # 初始化Bi-LSTM模型
        self.model = BiLSTMActionSegmentation(input_size=34, hidden_size=64, num_layers=2, num_classes=3)  # 假设有3个阶段
        self.model.eval()  # 设置为评估模式
        
        # 初始化DTW余弦相似度匹配器
        self.pose_matcher = DTWCosineMatcher(threshold=0.75)
        
        # 初始化相对L₂评分器
        self.pose_scorer = RelativeL2Scorer()
        
        # 存储阶段预测结果
        self.phase_predictions = []
        
        # 参考姿态数据路径
        self.reference_data_path = 'data/reference_poses.json'
        self.reference_keypoints = None
        self.reference_predictions = None
        
        # 加载参考姿态数据（如果存在）
        self.load_reference_data()
        
    def load_reference_data(self):
        """加载参考姿态数据"""
        try:
            if os.path.exists(self.reference_data_path):
                with open(self.reference_data_path, 'r') as f:
                    reference_data = json.load(f)
                    self.reference_keypoints = np.array(reference_data.get('keypoints', []))
                    self.reference_predictions = np.array(reference_data.get('predictions', []))
                    print(f"已加载参考姿态数据，共{len(self.reference_keypoints)}帧")
        except Exception as e:
            print(f"加载参考姿态数据失败: {e}")
            self.reference_keypoints = None
            self.reference_predictions = None
        
    def set_start_zone(self, x: int, y: int, width: int, height: int):
        """设置起跳预备区域"""
        self.start_zone = (x, y, width, height)
        
    def is_in_start_zone(self, bbox: np.ndarray) -> bool:
        """检查检测框是否在起跳预备区域内"""
        if self.start_zone is None:
            return False
            
        sx, sy, sw, sh = self.start_zone
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return (sx <= center_x <= sx + sw) and (sy <= center_y <= sy + sh)

    def process_frame(self, video_path, frame_id=0):
        """处理视频帧并返回结果"""
        cap = cv2.VideoCapture(video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        
        # 重置轨迹和预测数据
        self.trajectory = []
        self.phase_predictions = []
        
        # 批处理关键点，用于Bi-LSTM模型的输入
        keypoints_batch = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # 处理当前帧
                frame_info = {}
                
                # 1. 先获取人体检测框
                bboxes = self.human.get_bbox_one_image(frame)
                
                if len(bboxes) > 0:
                    # 2. 使用检测框获取姿态估计结果
                    pose_results = self.human.get_pose_one_image(frame, bboxes)
                    
                    if pose_results and len(pose_results) > 0:
                        # 获取第一个检测到的人的关键点
                        instance = pose_results[0]
                        keypoints = instance['keypoints']
                        keypoint_scores = instance.get('keypoint_scores', np.ones(len(keypoints)))  # 如果没有分数，使用默认值1
                        
                        # 确保keypoints是numpy数组
                        keypoints = np.array(keypoints)
                        keypoint_scores = np.array(keypoint_scores)
                        
                        # 转换为边界框
                        bbox = self.keypoints_to_bbox(keypoints)
                        
                        # 更新追踪器
                        if not self.tracking_active:
                            if self.is_in_start_zone(bbox):
                                self.tracking_active = True
                                self.target_id = len(self.trajectory)
                        
                        if self.tracking_active:
                            # 收集关键点用于批处理
                            keypoints_batch.append(keypoints)
                            
                            # 将关键点数据转换为模型输入格式
                            keypoints_input = self.prepare_keypoints_for_model(keypoints)
                            
                            # 使用Bi-LSTM模型进行阶段划分
                            with torch.no_grad():
                                outputs = self.model(torch.tensor(keypoints_input).unsqueeze(0))
                                _, predicted_phase = torch.max(outputs, 1)
                            
                            # 记录阶段信息
                            phase_id = predicted_phase.item()
                            self.phase_predictions.append(phase_id)
                            
                            # 记录关键点和分数
                            frame_info = {
                                'frame_id': self.current_frame,
                                'keypoints': keypoints.tolist(),
                                'scores': keypoint_scores.tolist(),
                                'bbox': bbox.tolist(),
                                'predicted_phase': phase_id,
                                'phase_name': ['takeoff', 'flight', 'landing'][phase_id] if phase_id < 3 else f'phase_{phase_id}'
                            }
                            self.trajectory.append(frame_info)
                            
                            # 在帧上绘制结果
                            self.draw_results(frame, bbox, keypoints, keypoint_scores, phase_id)
                
                # 更新进度
                self.current_frame += 1
                progress = int((self.current_frame / self.total_frames) * 25)  # 视频处理占总进度的25%
                
                # 返回进度信息
                yield progress, frame, frame_info
                
            except Exception as e:
                print(f"处理第 {self.current_frame} 帧时出错: {str(e)}")
                import traceback
                traceback.print_exc()  # 打印完整的错误堆栈
                continue
        
        # 视频处理完成后，执行姿态匹配和评估
        if self.tracking_active and self.reference_keypoints is not None:
            try:
                # 将关键点转换为numpy数组
                all_keypoints = np.array([frame_info['keypoints'] for frame_info in self.trajectory])
                all_predictions = np.array(self.phase_predictions)
                
                # 使用DTW和余弦相似度进行姿态匹配和评估
                alignment_result = self.pose_matcher.evaluate_performance(
                    self.reference_keypoints, 
                    all_keypoints, 
                    all_predictions
                )
                
                # 保存匹配结果
                self.save_alignment_results(alignment_result, os.path.join(os.path.dirname(video_path), 'alignment_result.json'))
                
                # 可视化匹配结果
                self.pose_matcher.visualize_alignment(
                    self.pose_matcher.segment_phases(self.pose_matcher.preprocess_keypoints(self.reference_keypoints), self.reference_predictions),
                    self.pose_matcher.segment_phases(self.pose_matcher.preprocess_keypoints(all_keypoints), all_predictions),
                    alignment_result,
                    os.path.join(os.path.dirname(video_path), 'alignment_plots')
                )
                
                # 生成姿态匹配报告
                self.pose_matcher.generate_alignment_report(
                    alignment_result,
                    os.path.join(os.path.dirname(video_path), 'alignment_report.json')
                )
                
                # 使用相对L₂距离进行动作评分
                self.evaluate_jump_performance(
                    all_keypoints, 
                    all_predictions,
                    os.path.join(os.path.dirname(video_path), 'score_result.json')
                )
                
            except Exception as e:
                print(f"执行姿态匹配和评分时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        
        cap.release()
    
    def evaluate_jump_performance(self, keypoints, predictions, output_path):
        """评估跳远表现并生成评分报告"""
        try:
            if self.reference_keypoints is None or len(self.reference_keypoints) == 0:
                print("未找到参考姿态数据，无法评分")
                return False
                
            # 预处理关键点数据
            target_keypoints = self.pose_matcher.preprocess_keypoints(keypoints)
            reference_keypoints = self.pose_matcher.preprocess_keypoints(self.reference_keypoints)
            
            # 分割动作阶段
            target_phases = self.pose_matcher.segment_phases(target_keypoints, predictions)
            reference_phases = self.pose_matcher.segment_phases(reference_keypoints, self.reference_predictions)
            
            # 计算相对L₂评分
            score_result = self.pose_scorer.score_jump_performance(target_phases, reference_phases)
            
            # 保存评分结果
            self.pose_scorer.save_score_report(score_result, output_path)
            
            print(f"\n评分结果: 总分 {score_result['overall_score']:.2f}")
            print(f"整体评价: {score_result['overall_evaluation']}")
            
            # 打印各阶段得分
            print("\n各阶段得分:")
            for phase in score_result['phase_evaluations']:
                print(f"- {phase['phase_name']}: {phase['score']:.2f} ({phase['evaluation']})")
            
            # 打印改进建议
            if score_result['suggestions']:
                print("\n改进建议:")
                for suggestion in score_result['suggestions']:
                    print(f"- {suggestion}")
            
            return True
            
        except Exception as e:
            print(f"评估跳远表现时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def draw_results(self, frame: np.ndarray, bbox: np.ndarray, keypoints: np.ndarray, keypoint_scores: np.ndarray, phase_id=None):
        """绘制检测框、骨架和关键点"""
        try:
            # 确保bbox是正确的格式
            bbox = np.array(bbox).astype(int)  # 转换为整数坐标
            if len(bbox) == 4:  # 确保有4个坐标点
                # 绘制检测框
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加ID和阶段信息
                info_text = f"ID: {self.target_id}"
                if phase_id is not None:
                    phase_names = ["起跳", "腾空", "收腹"]
                    if 0 <= phase_id < len(phase_names):
                        info_text += f" | {phase_names[phase_id]}"
                
                cv2.putText(frame, info_text, 
                           (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 将关键点数据转换为numpy数组并确保是整数
                keypoints = np.array(keypoints).astype(int)
                keypoint_scores = np.array(keypoint_scores)
                
                # 绘制骨架
                for connection in self.KEYPOINT_CONNECTIONS:
                    start_idx, end_idx = connection
                    if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                        keypoint_scores[start_idx] > 0.3 and keypoint_scores[end_idx] > 0.3):
                        start_point = tuple(keypoints[start_idx])
                        end_point = tuple(keypoints[end_idx])
                        if all(p >= 0 for p in start_point + end_point):  # 确保所有坐标都是非负的
                            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
                
                # 绘制关键点
                for idx, (kp, score) in enumerate(zip(keypoints, keypoint_scores)):
                    if score > 0.3:  # 置信度值
                        x, y = int(kp[0]), int(kp[1])
                        if x >= 0 and y >= 0:  # 确保坐标是非负的
                            color = (0, 0, 255)  # 默认红色
                            if idx in self.key_point_indices.values():
                                color = (0, 255, 0)  # 主要关节点用绿色
                            cv2.circle(frame, (x, y), 4, color, -1)
                            cv2.putText(frame, f"{idx}", (x-10, y-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                
        except Exception as e:
            print(f"绘制结果时出错: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印完整的错误堆栈

    def save_results(self, output_path: str):
        """
        保存分析结果
        Args:
            output_path: 保存路径
        """
        try:
            result_data = {
                'total_frames': len(self.trajectory),
                'tracking_info': {
                    'target_id': self.target_id,
                    'start_zone': self.start_zone
                },
                'trajectory': self.trajectory,
                'phase_predictions': self.phase_predictions
            }
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存为JSON文件
            with open(output_path, 'w') as f:
                json.dump(result_data, f, indent=4)
                
            print(f"\n分析结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"保存结果时出错: {str(e)}")
    
    def save_alignment_results(self, alignment_result, output_path):
        """保存姿态对齐结果"""
        try:
            # 转换numpy数组为列表，以便JSON序列化
            json_compatible_result = {}
            for key, value in alignment_result.items():
                if key == 'overall_similarity':
                    json_compatible_result[key] = float(value)
                else:
                    json_compatible_result[key] = {
                        'distance': float(value['distance']),
                        'path': [[int(i), int(j)] for i, j in value['path']],
                        'similarity': float(value['similarity']),
                        'reference_length': int(value['reference_length']),
                        'target_length': int(value['target_length'])
                    }
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存为JSON文件
            with open(output_path, 'w') as f:
                json.dump(json_compatible_result, f, indent=4)
                
            print(f"\n姿态对齐结果已保存到: {output_path}")
        except Exception as e:
            print(f"保存姿态对齐结果时出错: {str(e)}")

    def keypoints_to_bbox(self, keypoints):
        """将关键点转换为边界框"""
        try:
            if isinstance(keypoints, list):
                keypoints = np.array(keypoints)
            
            # 检查关键点数据的形状
            if len(keypoints.shape) == 2:  # 如果只有x,y坐标
                # 直接使用所有关键点
                x_coords = keypoints[:, 0]
                y_coords = keypoints[:, 1]
            else:  # 如果有置信度信息
                # 获取所有有效关键点（忽略置信度为0的点）
                valid_keypoints = keypoints[keypoints[:, 2] > 0]
                if len(valid_keypoints) == 0:
                    return np.array([0, 0, 100, 100])  # 返回一个默认的边界框
                x_coords = valid_keypoints[:, 0]
                y_coords = valid_keypoints[:, 1]
            
            # 计算边界框
            x1 = np.min(x_coords)
            y1 = np.min(y_coords)
            x2 = np.max(x_coords)
            y2 = np.max(y_coords)
            
            # 添加一些边距
            margin = 10
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = x2 + margin
            y2 = y2 + margin
            
            # 确保返回整数坐标
            return np.array([int(x1), int(y1), int(x2), int(y2)])
        
        except Exception as e:
            print(f"计算边界框时出错: {str(e)}")
            return np.array([0, 0, 100, 100])  # 返回一个默认的边界框

    def process_frame_for_display(self, frame):
        """处理单帧用于显示"""
        try:
            # 1. 获取人体检测框
            bboxes = self.human.get_bbox_one_image(frame)
            
            if len(bboxes) > 0:
                # 2. 使用检测框获取姿态估计结果
                pose_results = self.human.get_pose_one_image(frame, bboxes)
                
                if pose_results and len(pose_results) > 0:
                    # 获取第一个检测到的人的关键点
                    instance = pose_results[0]
                    keypoints = instance['keypoints']
                    keypoint_scores = instance.get('keypoint_scores', np.ones(len(keypoints)))
                    
                    # 确保keypoints是numpy数组
                    keypoints = np.array(keypoints)
                    keypoint_scores = np.array(keypoint_scores)
                    
                    # 转换为边界框
                    bbox = self.keypoints_to_bbox(keypoints)
                    
                    # 使用Bi-LSTM模型进行阶段预测
                    keypoints_input = self.prepare_keypoints_for_model(keypoints)
                    with torch.no_grad():
                        outputs = self.model(torch.tensor(keypoints_input).unsqueeze(0))
                        _, predicted_phase = torch.max(outputs, 1)
                    
                    # 在帧上绘制结果
                    frame_with_pose = frame.copy()
                    self.draw_results(frame_with_pose, bbox, keypoints, keypoint_scores, predicted_phase.item())
                    return frame_with_pose
            
            return frame
            
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
            return frame
    
    def prepare_keypoints_for_model(self, keypoints):
        """准备关键点数据以输入到Bi-LSTM模型"""
        # 假设关键点是一个17x2的数组，转换为34维的输入
        return keypoints.flatten()
    
    def save_as_reference(self, output_path=None):
        """将当前分析结果保存为参考姿态数据"""
        if not self.trajectory:
            print("没有可保存的轨迹数据")
            return False
            
        try:
            # 准备要保存的数据
            all_keypoints = [frame_info['keypoints'] for frame_info in self.trajectory]
            
            reference_data = {
                'keypoints': all_keypoints,
                'predictions': self.phase_predictions,
                'metadata': {
                    'total_frames': len(self.trajectory),
                    'date_created': str(datetime.datetime.now())
                }
            }
            
            # 确定保存路径
            if output_path is None:
                output_path = self.reference_data_path
                
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存为JSON文件
            with open(output_path, 'w') as f:
                json.dump(reference_data, f, indent=4)
                
            print(f"\n参考姿态数据已保存到: {output_path}")
            
            # 更新当前的参考数据
            self.reference_keypoints = np.array(all_keypoints)
            self.reference_predictions = np.array(self.phase_predictions)
            
            return True
            
        except Exception as e:
            print(f"保存参考姿态数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False