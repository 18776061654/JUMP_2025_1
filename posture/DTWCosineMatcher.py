import numpy as np
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import os
import json

class DTWCosineMatcher:
    def __init__(self, threshold=0.8):
        """
        初始化DTW和余弦相似度融合的姿态匹配器
        
        Args:
            threshold: 相似度阈值，低于此值的匹配被视为不匹配
        """
        self.threshold = threshold
        self.phase_names = {
            'takeoff': '起跳阶段',
            'flight': '腾空阶段',
            'landing': '收腹阶段'
        }
        
    def preprocess_keypoints(self, keypoints):
        """
        预处理关键点数据，包括归一化和特征提取
        
        Args:
            keypoints: 关键点数据，形状为 [frames, joints, coordinates]
            
        Returns:
            处理后的特征向量
        """
        # 确保输入数据是numpy数组
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)
        
        # 获取关键点数量和坐标维度
        n_frames, n_joints, n_coords = keypoints.shape
        
        # 1. 归一化坐标，使其在[0,1]范围内
        x_min, y_min = np.min(keypoints[:, :, 0]), np.min(keypoints[:, :, 1])
        x_max, y_max = np.max(keypoints[:, :, 0]), np.max(keypoints[:, :, 1])
        
        normalized_keypoints = keypoints.copy()
        normalized_keypoints[:, :, 0] = (keypoints[:, :, 0] - x_min) / (x_max - x_min + 1e-8)
        normalized_keypoints[:, :, 1] = (keypoints[:, :, 1] - y_min) / (y_max - y_min + 1e-8)
        
        # 2. 特征提取：将关键点展平为特征向量
        features = normalized_keypoints.reshape(n_frames, -1)
        
        return features
    
    def segment_phases(self, features, predictions):
        """
        将连续帧按预测的阶段进行分割
        
        Args:
            features: 关键点特征，形状为 [frames, features]
            predictions: 每一帧的阶段预测，形状为 [frames]
            
        Returns:
            按阶段分组的特征字典
        """
        phases = {}
        for phase_id in np.unique(predictions):
            # 获取属于当前阶段的所有帧索引
            phase_indices = np.where(predictions == phase_id)[0]
            
            # 如果帧数太少，跳过
            if len(phase_indices) < 3:
                continue
                
            # 获取这个阶段的所有帧特征
            phase_name = ['takeoff', 'flight', 'landing'][phase_id] if phase_id < 3 else f'phase_{phase_id}'
            phases[phase_name] = features[phase_indices]
            
        return phases
    
    def calculate_cosine_similarity(self, vec1, vec2):
        """
        计算两个向量之间的余弦相似度
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            余弦相似度，范围在[-1, 1]之间，值越大表示越相似
        """
        # 处理零向量情况
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0
            
        return 1 - cosine(vec1, vec2)  # 转换cosine距离为相似度

    def custom_distance(self, vec1, vec2):
        """
        自定义距离度量，结合余弦相似度和欧氏距离
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            组合距离，值越小表示越相似
        """
        # 余弦相似度（转换为距离）
        cos_sim = self.calculate_cosine_similarity(vec1, vec2)
        cos_dist = 1 - cos_sim
        
        # 归一化欧氏距离
        euclidean_dist = np.linalg.norm(vec1 - vec2) / (np.linalg.norm(vec1) + np.linalg.norm(vec2) + 1e-8)
        
        # 组合距离（权重可以调整）
        combined_dist = 0.7 * cos_dist + 0.3 * euclidean_dist
        
        return combined_dist

    def dtw_cosine_distance(self, sequence1, sequence2):
        """
        使用DTW和自定义距离度量计算两个序列之间的距离
        
        Args:
            sequence1: 第一个序列，形状为 [frames1, features]
            sequence2: 第二个序列，形状为 [frames2, features]
            
        Returns:
            距离值和路径
        """
        distance, path = fastdtw(sequence1, sequence2, dist=self.custom_distance)
        return distance, path

    def align_phases(self, reference_phases, target_phases):
        """
        对齐起跳、腾空和收腹三个阶段
        
        Args:
            reference_phases: 参考动作的各阶段特征
            target_phases: 目标动作的各阶段特征
            
        Returns:
            各阶段的对齐结果
        """
        aligned_phases = {}
        overall_similarity = 0
        for phase in ['takeoff', 'flight', 'landing']:
            if phase not in reference_phases or phase not in target_phases:
                print(f"警告：{phase}阶段在参考或目标数据中缺失")
                continue
                
            ref_seq = reference_phases[phase]
            target_seq = target_phases[phase]
            
            # 计算DTW距离和路径
            distance, path = self.dtw_cosine_distance(ref_seq, target_seq)
            
            # 计算序列平均相似度（1-distance/max_length）
            max_length = max(len(ref_seq), len(target_seq))
            similarity = max(0, 1 - distance / max_length)
            
            aligned_phases[phase] = {
                'distance': distance,
                'path': path,
                'similarity': similarity,
                'reference_length': len(ref_seq),
                'target_length': len(target_seq)
            }
            
            overall_similarity += similarity
            
        # 计算总体相似度
        if aligned_phases:
            overall_similarity /= len(aligned_phases)
            aligned_phases['overall_similarity'] = overall_similarity
            
        return aligned_phases
    
    def visualize_alignment(self, reference_phases, target_phases, aligned_phases, output_dir=None):
        """
        可视化对齐结果
        
        Args:
            reference_phases: 参考动作的各阶段特征
            target_phases: 目标动作的各阶段特征
            aligned_phases: 对齐结果
            output_dir: 输出目录，如果提供则保存图片
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.figure(figsize=(15, 10))
        
        for i, phase in enumerate(['takeoff', 'flight', 'landing']):
            if phase not in aligned_phases:
                continue
                
            plt.subplot(3, 1, i+1)
            
            ref_seq = reference_phases[phase]
            target_seq = target_phases[phase]
            path = aligned_phases[phase]['path']
            similarity = aligned_phases[phase]['similarity']
            
            # 将路径转换为索引对
            path_indices = np.array(path)
            
            # 绘制对齐路径
            plt.plot(path_indices[:, 0], path_indices[:, 1], 'r-', linewidth=2)
            
            # 添加标题和标签
            plt.title(f"{self.phase_names.get(phase, phase)} - 相似度: {similarity:.2f}")
            plt.xlabel('参考序列帧')
            plt.ylabel('目标序列帧')
            plt.grid(True)
            
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'phase_alignment.png'))
            plt.close()
        else:
            plt.show()
    
    def generate_alignment_report(self, aligned_phases, output_path=None):
        """
        生成对齐报告
        
        Args:
            aligned_phases: 对齐结果
            output_path: 输出文件路径，如果提供则保存为JSON
            
        Returns:
            报告字典
        """
        report = {
            "overall_similarity": aligned_phases.get('overall_similarity', 0),
            "phases": {}
        }
        
        for phase in ['takeoff', 'flight', 'landing']:
            if phase not in aligned_phases:
                continue
                
            phase_data = aligned_phases[phase]
            report["phases"][phase] = {
                "similarity": phase_data['similarity'],
                "reference_length": phase_data['reference_length'],
                "target_length": phase_data['target_length'],
                "is_matched": phase_data['similarity'] >= self.threshold
            }
            
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
                
        return report
    
    def evaluate_performance(self, reference_keypoints, target_keypoints, predictions):
        """
        评估目标动作与参考动作的匹配程度
        
        Args:
            reference_keypoints: 参考动作的关键点序列 [frames, joints, coords]
            target_keypoints: 目标动作的关键点序列 [frames, joints, coords]
            predictions: 每一帧的阶段预测 [frames]
            
        Returns:
            评估结果
        """
        # 预处理关键点
        ref_features = self.preprocess_keypoints(reference_keypoints)
        target_features = self.preprocess_keypoints(target_keypoints)
        
        # 分割阶段
        ref_phases = self.segment_phases(ref_features, predictions)
        target_phases = self.segment_phases(target_features, predictions)
        
        # 对齐阶段
        aligned_phases = self.align_phases(ref_phases, target_phases)
        
        return aligned_phases