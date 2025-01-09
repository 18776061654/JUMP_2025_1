import json
import numpy as np
from typing import List, Dict, Tuple
import os
from scipy.spatial.distance import cosine
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import tkinter as tk
from tkinter import filedialog
import cv2
from datetime import datetime

class PoseMatcher:
    def __init__(self):
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
        
        # 扩展关节角度组合，增加更多特征点组合
        self.joint_angles = [
            # 躯干角度
            ('left_shoulder', 'left_hip', 'left_knee'),    # 左侧躯干角度
            ('right_shoulder', 'right_hip', 'right_knee'), # 右侧躯干角度
            ('left_hip', 'right_hip', 'left_knee'),       # 左髋部角度
            ('right_hip', 'left_hip', 'right_knee'),      # 右髋部角度
            
            # 手臂角度
            ('left_shoulder', 'left_elbow', 'left_wrist'),    # 左臂角度
            ('right_shoulder', 'right_elbow', 'right_wrist'), # 右臂角度
            
            # 腿部角度
            ('left_hip', 'left_knee', 'left_ankle'),    # 左腿角度
            ('right_hip', 'right_knee', 'right_ankle'), # 右腿角度
        ]
        
        # 关键点对，用于计算相对距离
        self.keypoint_pairs = [
            ('left_shoulder', 'right_shoulder'),  # 肩宽
            ('left_hip', 'right_hip'),           # 髋宽
            ('left_knee', 'right_knee'),         # 膝距
            ('left_ankle', 'right_ankle'),       # 踝距
        ]
        
        # 特征权重
        self.weights = {
            'angles': 0.6,      # 角度特征权重
            'distances': 0.2,   # 距离特征权重
            'positions': 0.2    # 位置特征权重
        }
        
        # 添加角度计算的权重配置
        self.angle_weights = {
            'trunk': 0.4,      # 躯干角度权重
            'arms': 0.3,       # 手臂角度权重
            'legs': 0.3        # 腿部角度权重
        }
        
        # 余弦距离的权重系数
        self.distance_coefficients = {
            'euclidean': 0.7,  # 欧氏距离权重
            'cosine': 0.3      # 余弦距离权重
        }

    def load_standard_pose(self, standard_pose_path: str) -> Dict:
        """加载标准姿态数据"""
        try:
            with open(standard_pose_path, 'r') as f:
                data = json.load(f)
            
            # 打印数据结构以便调试
            print(f"标准姿态文件内容: {data.keys() if isinstance(data, dict) else type(data)}")
            
            # 确保关键点数据是正确的格式
            if isinstance(data, dict) and 'keypoints' in data:
                keypoints_data = data['keypoints']
                if 'coordinates' in keypoints_data:
                    # 从coordinates字段获取关键点坐标
                    keypoints = np.array(keypoints_data['coordinates'])
                    scores = np.array(keypoints_data.get('scores', [1.0] * len(keypoints)))
                else:
                    raise ValueError("标准姿态文件缺少coordinates字段")
            else:
                raise ValueError("标准姿态文件格式错误")
            
 
            
            # 确保关键点数组具有正确的维度
            if len(keypoints.shape) != 2 or keypoints.shape[1] != 2:
                raise ValueError(f"关键点数组维度错误: {keypoints.shape}, 应为 (N, 2)")
            
            return {
                'pose_results': [{
                    'keypoints': keypoints,
                    'scores': scores
                }]
            }
        except Exception as e:
            print(f"加载标准姿态数据出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def load_test_pose(self, test_pose_path: str) -> Dict:
        """
        加载待测试姿态数据，适配新的数据格式
        """
        try:
            with open(test_pose_path, 'r') as f:
                test_data = json.load(f)
            
            # 检查新的数据格式
            if 'data' in test_data and 'frames' in test_data['data']:
                # 转换为期望的格式
                return {
                    'trajectory': [
                        {
                            'frame_idx': frame['frame_idx'],
                            'actual_frame': frame['frame_idx'],
                            'keypoints': frame['keypoints'],
                            'keypoint_scores': frame.get('scores', [1.0] * len(frame['keypoints']))
                        }
                        for frame in test_data['data']['frames']
                    ]
                }
            elif 'trajectory' in test_data:
                # 如果已经是trajectory格式，直接返回
                return test_data
            elif isinstance(test_data, list):
                # 如果是关键点列表格式
                return {
                    'trajectory': [{
                        'frame_idx': 0,
                        'actual_frame': 0,
                        'keypoints': test_data,
                        'keypoint_scores': [1.0] * len(test_data)
                    }]
                }
            else:
                print(f"数据格式: {test_data.keys() if isinstance(test_data, dict) else type(test_data)}")
                raise ValueError("不支持的数据格式")
                
        except Exception as e:
            print(f"加载待测试姿态数据出错: {str(e)}")
            # 打印更详细的错误信息
            import traceback
            traceback.print_exc()
            return None

    def calculate_angle(self, point1: np.ndarray, point2: np.ndarray, 
                       point3: np.ndarray) -> float:
        """使用改进的角度计算方法"""
        try:
            # 转换为弧度
            vector1 = point1 - point2
            vector2 = point3 - point2
            
            # 计算点积
            dot_product = np.dot(vector1, vector2)
            # 计算向量的模
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            # 计算角度（弧度）
            cos_angle = dot_product / (norm1 * norm2)
            # 处理数值误差
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # 转换为角度
            angle_degrees = np.degrees(angle)
            
            # 检查角度是否接近于零
            if abs(angle_degrees) < 1e-6:
                angle_degrees = 0.0
            
            return angle_degrees
        except Exception as e:
            print(f"计算角度时出错: {str(e)}")
            return 0.0

    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """归一化关键点坐标"""
        try:
            # 确保输入是numpy数组
            if not isinstance(keypoints, np.ndarray):
                keypoints = np.array(keypoints)
            
            # 确保关键点数组具有正确的维度
            if len(keypoints.shape) == 1:
                keypoints = keypoints.reshape(-1, 2)
            elif len(keypoints.shape) == 3:
                keypoints = keypoints.squeeze()
            
            # 检查关键点数组的形状
            if keypoints.shape[0] < max(self.key_point_indices.values()) + 1:
                raise ValueError(f"关键点数量不足: {keypoints.shape[0]}")
            
            # 找到躯干关键点（肩部和髋部）
            shoulders = np.array([
                keypoints[self.key_point_indices['left_shoulder']][:2],
                keypoints[self.key_point_indices['right_shoulder']][:2]
            ])
            hips = np.array([
                keypoints[self.key_point_indices['left_hip']][:2],
                keypoints[self.key_point_indices['right_hip']][:2]
            ])
            
            # 计算躯干中心点
            center = np.mean(np.vstack([shoulders, hips]), axis=0)
            
            # 计算缩放因子（使用躯干长度）
            torso_length = np.linalg.norm(np.mean(shoulders, axis=0) - np.mean(hips, axis=0))
            if torso_length == 0:
                torso_length = 1.0  # 防止除以零
            
            # 归一化所有关键点
            normalized = np.array(keypoints.copy())
            normalized[:, :2] = (normalized[:, :2] - center) / torso_length
            
            return normalized
            
        except Exception as e:
            print(f"归一化关键点时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return keypoints

    def extract_pose_features(self, keypoints: np.ndarray, keypoint_scores: np.ndarray = None) -> Dict:
        """提取增强的姿态特征"""
        try:
            # 数据验证
            if not isinstance(keypoints, np.ndarray):
                keypoints = np.array(keypoints)
            
            # 确保关键点数组具有正确的维度
            if len(keypoints.shape) == 1:
                keypoints = keypoints.reshape(-1, 2)
            elif len(keypoints.shape) == 3:
                keypoints = keypoints.squeeze()
            
         
            
            # 归一化关键点
            norm_keypoints = self.normalize_keypoints(keypoints)
            
            features = {
                'angles': [],
                'distances': [],
                'positions': [],
                'scores': []
            }
            
            # 确保关键点索引在有效范围内
            max_index = max(self.key_point_indices.values())
            if norm_keypoints.shape[0] <= max_index:
                raise ValueError(f"关键点数量不足: {norm_keypoints.shape[0]}, 需要: {max_index + 1}")
            
            # 1. 计算角度特征
            for joint_combo in self.joint_angles:
                p1_idx = self.key_point_indices[joint_combo[0]]
                p2_idx = self.key_point_indices[joint_combo[1]]
                p3_idx = self.key_point_indices[joint_combo[2]]
                
                p1 = norm_keypoints[p1_idx][:2]
                p2 = norm_keypoints[p2_idx][:2]
                p3 = norm_keypoints[p3_idx][:2]
                
                angle = self.calculate_angle(p1, p2, p3)
                features['angles'].append(angle)
                
                # 添加对应的置信度
                if keypoint_scores is not None:
                    score = np.mean([keypoint_scores[p1_idx], 
                                   keypoint_scores[p2_idx], 
                                   keypoint_scores[p3_idx]])
                    features['scores'].append(score)
            
            # 2. 计算相对距离特征
            for pair in self.keypoint_pairs:
                idx1 = self.key_point_indices[pair[0]]
                idx2 = self.key_point_indices[pair[1]]
                
                p1 = norm_keypoints[idx1][:2]
                p2 = norm_keypoints[idx2][:2]
                
                distance = np.linalg.norm(p1 - p2)
                features['distances'].append(distance)
            
            # 3. 添加归一化位置特征
            for key_point in self.key_point_indices.values():
                features['positions'].extend(norm_keypoints[key_point][:2])
            
            return features
            
        except Exception as e:
            print(f"提取姿态特征时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_pose_similarity(self, standard_features: Dict, test_features: Dict) -> float:
        """使用改进的相似度计算方法"""
        try:
            if not standard_features or not test_features:
                return 0.0
            
            # 1. 计算角度的余弦相似度
            angles_std = np.array(standard_features['angles'])
            angles_test = np.array(test_features['angles'])
            
            # 将角度转换为弧度
            rad_std = np.radians(angles_std)
            rad_test = np.radians(angles_test)
            
            # 计算余弦和正弦值
            cos_std = np.cos(rad_std)
            sin_std = np.sin(rad_std)
            cos_test = np.cos(rad_test)
            sin_test = np.sin(rad_test)
            
            # 计算点积和模长
            dot_products = cos_std * cos_test + sin_std * sin_test
            magnitudes_std = np.sqrt(cos_std**2 + sin_std**2)
            magnitudes_test = np.sqrt(cos_test**2 + sin_test**2)
            
            # 计算余弦相似度
            cosine_similarities = dot_products / (magnitudes_std * magnitudes_test)
            cosine_distance = 1 - np.mean(cosine_similarities)
            
            # 2. 计算欧氏距离
            euclidean_distance = np.sqrt(np.mean((angles_std - angles_test) ** 2))
            normalized_euclidean = euclidean_distance / 180.0  # 归一化到 [0,1] 范围
            
            # 3. 组合两种距离度量
            similarity = (1 - (
                self.distance_coefficients['euclidean'] * normalized_euclidean +
                self.distance_coefficients['cosine'] * cosine_distance
            ))
            
            # 4. 应用置信度权重（如果有）
            if test_features.get('scores'):
                confidence = np.mean(test_features['scores'])
                similarity *= confidence
            
            return max(0, min(1, similarity))  # 确保相似度在[0,1]范围内
            
        except Exception as e:
            print(f"计算姿态相似度出错: {str(e)}")
            return 0.0

    def find_best_matches(self, standard_pose_path: str, test_pose_path: str, 
                         threshold: float = 0.8) -> List[Dict]:
        """找出最佳匹配姿���帧"""
        try:
            # 加载数据
            standard_data = self.load_standard_pose(standard_pose_path)
            test_data = self.load_test_pose(test_pose_path)
            
            if not standard_data or not test_data:
                return []
            
            # 获取标准姿态特征
            standard_keypoints = np.array(standard_data['pose_results'][0]['keypoints'])
            standard_features = self.extract_pose_features(standard_keypoints)
            
            matches = []
            # 遍历测试视频中的每一帧
            for frame in test_data['trajectory']:
                test_keypoints = np.array(frame['keypoints'])
                test_scores = np.array(frame.get('keypoint_scores', [1.0] * len(test_keypoints)))
                test_features = self.extract_pose_features(test_keypoints, test_scores)
                
                # 计算相似度
                similarity = self.calculate_pose_similarity(standard_features, test_features)
                
                # 记录所有帧的匹配结果
                matches.append({
                    'frame_idx': frame.get('frame_idx', 0),
                    'actual_frame': frame.get('actual_frame', 0),
                    'similarity': float(similarity),
                    'keypoints': frame['keypoints'],
                    'keypoint_scores': test_scores.tolist()
                })
            
            # 按相似度降序排序
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 如果有超过阈值的匹配，只返回那些匹配
            above_threshold = [m for m in matches if m['similarity'] >= threshold]
            if above_threshold:
                return above_threshold
            
            # 如果没有超过阈值的匹配，返回相似度最高的一个
            return matches[:1] if matches else []
            
        except Exception as e:
            print(f"查找最佳匹配时出错: {str(e)}")
            return []

    def save_matches(self, matches: List[Dict], output_path: str, stage_id: str = None):
        """
        保存匹配结果，增加阶段信息
        """
        try:
            # 准备保存的数据
            result_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_matches': len(matches),
                'stage_id': stage_id,
                'best_match': matches[0] if matches else None,
                'match_statistics': {
                    'average_similarity': sum(m['similarity'] for m in matches) / len(matches) if matches else 0,
                    'max_similarity': max(m['similarity'] for m in matches) if matches else 0,
                    'min_similarity': min(m['similarity'] for m in matches) if matches else 0
                },
                'matches': matches
            }
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存为JSON文件
            with open(output_path, 'w') as f:
                json.dump(result_data, f, indent=4)
                
            print(f"\n匹配结果已保存到: {output_path}")
            print(f"找到 {len(matches)} 个匹配结果")
            
            # 打印最佳匹配信息
            if matches:
                best_match = matches[0]
                print(f"\n最佳匹配:")
                print(f"帧索引: {best_match['actual_frame']}")
                print(f"相似度: {best_match['similarity']:.3f}")
                print(f"平均相似度: {result_data['match_statistics']['average_similarity']:.3f}")
                    
        except Exception as e:
            print(f"保存匹配结果时出错: {str(e)}")

    def save_matched_frames(self, video_path, matches, output_dir, prefix=''):
        """保存匹配的帧和裁剪后的人体图像
        Args:
            video_path: 视频文件路径
            matches: 匹配结果
            output_dir: 输出目录
            prefix: 文件名前缀（可选）
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return
        
        try:
            # 处理matches是列表的情况
            if isinstance(matches, list) and matches:
                # 保存最佳匹配帧（列表中的第一个）
                best_match = matches[0]
                frame_idx = best_match.get('actual_frame', best_match.get('frame_idx'))
                bbox = best_match.get('bbox')  # 获取bbox信息
                
                if frame_idx is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        # 保存原始帧（带检测框）
                        frame_with_box = frame.copy()
                        if bbox:
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        output_path = os.path.join(output_dir, f'{prefix}frame_best.jpg')
                        cv2.imwrite(output_path, frame_with_box)
                        
                        # 保存裁剪后的人体图像
                        if bbox:
                            x1, y1, x2, y2 = map(int, bbox)
                            # 添加一些边距
                            margin = 10
                            h, w = frame.shape[:2]
                            x1 = max(0, x1 - margin)
                            y1 = max(0, y1 - margin)
                            x2 = min(w, x2 + margin)
                            y2 = min(h, y2 + margin)
                            cropped_frame = frame[y1:y2, x1:x2]
                            
                            # 保存裁剪图像
                            crop_output_path = os.path.join(output_dir, f'{prefix}frame_best_cropped.jpg')
                            cv2.imwrite(crop_output_path, cropped_frame)
                            print(f"已保存裁剪图像: {crop_output_path}")
                
                # 保存其他匹配帧
                for i, match in enumerate(matches[1:], 1):
                    frame_idx = match.get('actual_frame', match.get('frame_idx'))
                    bbox = match.get('bbox')
                    
                    if frame_idx is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if ret:
                            # 保存原始帧（带检测框）
                            frame_with_box = frame.copy()
                            if bbox:
                                x1, y1, x2, y2 = map(int, bbox)
                                cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            output_path = os.path.join(output_dir, f'{prefix}frame_{i}.jpg')
                            cv2.imwrite(output_path, frame_with_box)
                            
                            # 保存裁剪后的人体图像
                            if bbox:
                                x1, y1, x2, y2 = map(int, bbox)
                                # 添加一些边距
                                margin = 10
                                h, w = frame.shape[:2]
                                x1 = max(0, x1 - margin)
                                y1 = max(0, y1 - margin)
                                x2 = min(w, x2 + margin)
                                y2 = min(h, y2 + margin)
                                cropped_frame = frame[y1:y2, x1:x2]
                                
                                # 保存裁剪图像
                                crop_output_path = os.path.join(output_dir, f'{prefix}frame_{i}_cropped.jpg')
                                cv2.imwrite(crop_output_path, cropped_frame)
                                print(f"已保存裁剪图像: {crop_output_path}")
            
        finally:
            cap.release()

    def draw_skeleton(self, frame: np.ndarray, keypoints: List):
        """
        在图像上绘制骨架
        Args:
            frame: 图像帧
            keypoints: 关键点列表
        """
        try:
            keypoints = np.array(keypoints)
            
            # 绘制骨架连接
            for connection in self.joint_angles:
                idx1 = self.key_point_indices[connection[0]]
                idx2 = self.key_point_indices[connection[1]]
                
                if (idx1 < len(keypoints) and idx2 < len(keypoints)):
                    pt1 = tuple(map(int, keypoints[idx1][:2]))
                    pt2 = tuple(map(int, keypoints[idx2][:2]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            # 绘制关键点
            for idx, kp in enumerate(keypoints):
                if idx in self.key_point_indices.values():
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                    
        except Exception as e:
            print(f"绘制骨架时出错: {str(e)}")

    def load_stage_info(self, stage_file_path: str) -> Dict:
        """
        加载阶段划分信息
        Args:
            stage_file_path: 阶段划分JSON文件路径
        Returns:
            阶段信息字典
        """
        try:
            with open(stage_file_path, 'r', encoding='utf-8') as f:
                stage_data = json.load(f)
            return stage_data
        except Exception as e:
            print(f"加载阶段划分信息出错: {str(e)}")
            return None

    def find_best_matches_in_stage(self, standard_pose_path: str, test_pose_path: str, 
                                 stage_file_path: str, stage_id: str,
                                 threshold: float = 0.8) -> List[Dict]:
        """
        在指定阶段范围内找出最佳匹配的姿态帧
        Args:
            standard_pose_path: 标准姿态文件路径
            test_pose_path: 测试姿态文件路径
            stage_file_path: 阶段划分文件路径
            stage_id: 要匹配的阶段ID
            threshold: 相似度阈值
        Returns:
            匹配结果列表
        """
        try:
            # 加载数据
            standard_data = self.load_standard_pose(standard_pose_path)
            test_data = self.load_test_pose(test_pose_path)
            stage_data = self.load_stage_info(stage_file_path)
            
            # 打印详细的数据结构
            print("\n数据加载结果:")
            print(f"标准姿态数据: {standard_data['pose_results'][0]['keypoints'].shape if standard_data else 'None'}")
            print(f"测试数据轨迹长度: {len(test_data['trajectory']) if test_data else 'None'}")
            print(f"阶段数据: {stage_data['stages'] if stage_data else 'None'}")
            
            # 添加数据验证
            if not standard_data:
                print("无法加载标准姿态数据")
                return []
            if not test_data:
                print("无法加载测试姿态数据")
                return []
            if not stage_data:
                print("无法加载阶段数据")
                return []
            
            # 打印数据结构以便调试
            print(f"\n标准姿态数据结构: {standard_data.keys() if isinstance(standard_data, dict) else type(standard_data)}")
            print(f"测试姿态数据结构: {test_data.keys() if isinstance(test_data, dict) else type(test_data)}")
            print(f"阶段数据结构: {stage_data.keys() if isinstance(stage_data, dict) else type(stage_data)}")
            
            # 获取指定阶段的范围
            stage_info = None
            for stage in stage_data['stages']:
                if stage['id'] == stage_id:
                    stage_info = stage['info']
                    break
            
            if not stage_info:
                print(f"未找到定的阶段: {stage_id}")
                return []
            
            # 获取标准姿态特征
            standard_keypoints = np.array(standard_data['pose_results'][0]['keypoints'])
            standard_features = self.extract_pose_features(standard_keypoints)
            
            matches = []
            # 遍历测试视频中的每一帧
            for frame in test_data['trajectory']:
                # 获取当前帧的关键点
                keypoints = np.array(frame['keypoints'])
                
                # 计算人体中心点的x坐标（使用髋部中心点）
                left_hip = keypoints[self.key_point_indices['left_hip']]
                right_hip = keypoints[self.key_point_indices['right_hip']]
                center_x = (left_hip[0] + right_hip[0]) / 2
                
                # 归一化x坐标到[0,1]范围
                image_width = stage_data['image_size']['width']
                normalized_x = center_x / image_width
                
                # 检查是否在指定阶段范围内
                if stage_info['start'] <= normalized_x <= stage_info['end']:
                    test_scores = np.array(frame.get('keypoint_scores', [1.0] * len(keypoints)))
                    test_features = self.extract_pose_features(keypoints, test_scores)
                    
                    # 计算相似度
                    similarity = self.calculate_pose_similarity(standard_features, test_features)
                    
                    # 从关键点计算bbox
                    bbox = self.keypoints_to_bbox(keypoints)
                    
                    # 记录匹配结果
                    matches.append({
                        'frame_idx': frame['frame_idx'],
                        'actual_frame': frame.get('actual_frame', frame['frame_idx']),
                        'similarity': float(similarity),
                        'keypoints': frame['keypoints'],
                        'keypoint_scores': test_scores.tolist(),
                        'bbox': bbox.tolist(),  # 使用计算得到的bbox
                        'stage_id': stage_id,
                        'stage_name': stage_info.get('name', stage_id),
                        'position_x': normalized_x,
                        'center_x': center_x
                    })
            
            # 按相似度降序排序
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 如果有超过阈值的匹配，只返回那些匹配
            above_threshold = [m for m in matches if m['similarity'] >= threshold]
            if above_threshold:
                return above_threshold
            
            # 如果没有超过��值的匹配，返回相似度最高的一个
            return matches[:1] if matches else []
            
        except Exception as e:
            print(f"在阶段内查找最佳匹配时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

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
                valid_keypoints = keypoints[keypoints[:, 2] > 0] if keypoints.shape[1] > 2 else keypoints
                if len(valid_keypoints) == 0:
                    return np.array([0, 0, 100, 100])  # 返回一个默认的边界框
                x_coords = valid_keypoints[:, 0]
                y_coords = valid_keypoints[:, 1]
            
            # 计算边界框，添加边距
            margin = 20  # 增加边距使框更大一些
            x1 = max(0, np.min(x_coords) - margin)
            y1 = max(0, np.min(y_coords) - margin)
            x2 = np.max(x_coords) + margin
            y2 = np.max(y_coords) + margin
            
            # 确保返回整数坐标
            return np.array([int(x1), int(y1), int(x2), int(y2)])
            
        except Exception as e:
            print(f"计算边界框时出错: {str(e)}")
            return np.array([0, 0, 100, 100])  # 返回一个默认的边界框

def main():
    """主函数"""
    # 创建Tkinter根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 创建姿态匹配器
    matcher = PoseMatcher()
    
    # 选择视频文件
    print("\n请选择原始视频文件...")
    video_path = filedialog.askopenfilename(
        title="选择视频文件",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov"),
            ("All files", "*.*")
        ]
    )
    
    if not video_path:
        print("未选择视频文件")
        return
    
    # 选择标准姿态文件
    print("\n请选择标准姿态文件...")
    standard_pose_path = filedialog.askopenfilename(
        title="选择标准姿态文件",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    
    if not standard_pose_path:
        print("未选择标准姿态文件")
        return
        
    # 选择待测试姿态文件
    print("\n请选择待测试姿态文件...")
    test_pose_path = filedialog.askopenfilename(
        title="选择待测试姿态文件",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    
    if not test_pose_path:
        print("未选择待测试姿态文件")
        return
    
    # 选择保存结果的位置
    print("\n请选择保存结果的位置...")
    output_path = filedialog.asksaveasfilename(
        title="选择保存位置",
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    
    if not output_path:
        print("未选择保存位置")
        return
    
    print("\n开始姿态匹配分析...")
    print(f"视频文件: {video_path}")
    print(f"标准姿态文件: {standard_pose_path}")
    print(f"测试姿态文件: {test_pose_path}")
    print(f"结果将保存到: {output_path}")
    
    # 查找匹配
    matches = matcher.find_best_matches(standard_pose_path, test_pose_path)
    
    if matches:
        # 保存匹配结果JSON
        matcher.save_matches(matches, output_path)
        
        # 创建帧图像保存目录
        frames_dir = os.path.join(os.path.dirname(output_path), 'matched_frames')
        
        # 保存匹配帧
        saved_frames = matcher.save_matched_frames(video_path, matches, frames_dir)
        
        if saved_frames:
            print("\n已保存匹配帧图像:")
            for frame_path in saved_frames:
                print(f"- {frame_path}")
    else:
        print("\n未找到匹配的姿态")

if __name__ == "__main__":
    main()