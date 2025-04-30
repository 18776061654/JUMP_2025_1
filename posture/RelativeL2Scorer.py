import numpy as np
import os
import json
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import pandas as pd

class RelativeL2Scorer:
    def __init__(self, phase_weights=None):
        """初始化相对L₂距离评分器
        
        phase_weights: 各阶段权重字典，默认为均等权重
        """
        # 默认各阶段权重
        self.phase_weights = phase_weights or {
            'takeoff': 0.35,   # 起跳阶段权重
            'flight': 0.40,    # 腾空阶段权重
            'landing': 0.25    # 收腹阶段权重
        }
        
        # 相对L₂评分阈值参数
        self.score_thresholds = {
            'excellent': 0.85,  # 优秀阈值
            'good': 0.70,       # 良好阈值
            'average': 0.50,    # 一般阈值
            'poor': 0.30        # 较差阈值
        }
        
        # 评价文字模板
        self.evaluation_templates = {
            'excellent': '姿态标准，动作协调性优秀',
            'good': '姿态较好，动作协调性良好',
            'average': '姿态一般，动作协调性尚可',
            'poor': '姿态不足，需要改进动作协调性',
            'very_poor': '姿态欠佳，需要重点改进基本动作'
        }
        
        # 运动阶段中文名称映射
        self.phase_names = {
            'takeoff': '起跳阶段',
            'flight': '腾空阶段',
            'landing': '收腹阶段'
        }
    
    def calculate_relative_l2_distance(self, target_keypoints, reference_keypoints):
        """计算目标姿态与参考姿态之间的相对L₂距离
        
        R-L2 = (1/n)∑[(|s_i - s̄_i|)/(s_max - s_min)]²
        
        其中：
        s_i: 目标姿态的关键点值
        s̄_i: 参考姿态的关键点值
        s_max, s_min: 该运动类型可能出现的最大和最小得分范围
        n: 关键点数量
        """
        # 确保输入为numpy数组
        if not isinstance(target_keypoints, np.ndarray):
            target_keypoints = np.array(target_keypoints)
        if not isinstance(reference_keypoints, np.ndarray):
            reference_keypoints = np.array(reference_keypoints)
        
        # 计算关键点差异绝对值
        absolute_diff = np.abs(target_keypoints - reference_keypoints)
        
        # 计算关键点范围
        keypoints_max = np.max(reference_keypoints, axis=0) + 0.1  # 添加余量
        keypoints_min = np.min(reference_keypoints, axis=0) - 0.1  # 添加余量
        keypoints_range = keypoints_max - keypoints_min
        
        # 避免除零错误
        keypoints_range = np.where(keypoints_range < 1e-6, 1.0, keypoints_range)
        
        # 归一化差异
        normalized_diff = absolute_diff / keypoints_range
        
        # 计算相对L₂距离
        relative_l2 = np.mean(np.square(normalized_diff))
        
        # 转换为0-1之间的评分（距离越小，得分越高）
        score = np.exp(-5 * relative_l2)  # 使用指数衰减转换
        score = max(0.0, min(1.0, score))  # 确保在[0,1]范围内
        
        return score
    
    def evaluate_phase(self, phase_keypoints, reference_phase_keypoints):
        """评估单个阶段的姿态得分"""
        # 计算相对L₂距离评分
        score = self.calculate_relative_l2_distance(phase_keypoints, reference_phase_keypoints)
        
        # 确定评价等级
        if score >= self.score_thresholds['excellent']:
            evaluation = 'excellent'
        elif score >= self.score_thresholds['good']:
            evaluation = 'good'
        elif score >= self.score_thresholds['average']:
            evaluation = 'average'
        elif score >= self.score_thresholds['poor']:
            evaluation = 'poor'
        else:
            evaluation = 'very_poor'
        
        return {
            'score': score,
            'evaluation': self.evaluation_templates[evaluation]
        }
    
    def calculate_weighted_score(self, phase_scores):
        """计算加权总分"""
        total_score = 0.0
        total_weight = 0.0
        
        for phase, score in phase_scores.items():
            if phase in self.phase_weights:
                weight = self.phase_weights[phase]
                total_score += score * weight
                total_weight += weight
        
        # 防止权重总和为0
        if total_weight > 0:
            return total_score / total_weight
        return 0.0
    
    def generate_improvement_suggestions(self, phase_scores):
        """根据各阶段评分生成改进建议"""
        suggestions = []
        
        # 找出得分最低的阶段
        min_phase = min(phase_scores.items(), key=lambda x: x[1])
        
        if min_phase[1] < self.score_thresholds['average']:
            phase_name = self.phase_names.get(min_phase[0], min_phase[0])
            suggestions.append(f"建议重点改进{phase_name}的动作技术")
            
            if min_phase[0] == 'takeoff':
                suggestions.append("起跳阶段需要注意身体重心前移，髋关节充分伸展")
            elif min_phase[0] == 'flight':
                suggestions.append("腾空阶段需保持身体平衡，关注上下肢协调性")
            elif min_phase[0] == 'landing':
                suggestions.append("收腹阶段注意缓冲落地冲击，膝关节屈曲角度适中")
        
        # 整体建议
        if sum(phase_scores.values()) / len(phase_scores) < self.score_thresholds['good']:
            suggestions.append("整体动作流畅度有待提高，建议加强基本姿态训练")
        
        return suggestions
    
    def score_jump_performance(self, target_phases, reference_phases):
        """评分跳远表现
        
        target_phases: 目标动作各阶段关键点
        reference_phases: 参考动作各阶段关键点
        """
        result = {
            'phase_scores': {},
            'phase_evaluations': [],
            'overall_score': 0.0,
            'overall_evaluation': '',
            'suggestions': []
        }
        
        # 评估各阶段
        phase_scores = {}
        for phase in ['takeoff', 'flight', 'landing']:
            if phase in target_phases and phase in reference_phases:
                phase_eval = self.evaluate_phase(target_phases[phase], reference_phases[phase])
                phase_scores[phase] = phase_eval['score']
                
                # 记录阶段评价
                result['phase_evaluations'].append({
                    'phase': phase,
                    'phase_name': self.phase_names.get(phase, phase),
                    'score': phase_eval['score'],
                    'evaluation': phase_eval['evaluation'],
                    'weight': self.phase_weights.get(phase, 0)
                })
                
                # 记录阶段得分
                result['phase_scores'][phase] = {
                    'score': phase_eval['score'],
                    'weight': self.phase_weights.get(phase, 0),
                    'weighted_score': phase_eval['score'] * self.phase_weights.get(phase, 0)
                }
        
        # 计算加权总分
        overall_score = self.calculate_weighted_score(phase_scores)
        result['overall_score'] = overall_score
        
        # 总体评价
        if overall_score >= self.score_thresholds['excellent']:
            result['overall_evaluation'] = "整体表现优秀，姿态规范，动作协调性高"
        elif overall_score >= self.score_thresholds['good']:
            result['overall_evaluation'] = "整体表现良好，姿态基本规范，动作较为协调"
        elif overall_score >= self.score_thresholds['average']:
            result['overall_evaluation'] = "整体表现一般，姿态有待改进，动作协调性中等"
        elif overall_score >= self.score_thresholds['poor']:
            result['overall_evaluation'] = "整体表现不足，姿态需要调整，动作协调性较差"
        else:
            result['overall_evaluation'] = "整体表现欠佳，姿态与标准差距较大，需重点训练"
        
        # 生成改进建议
        result['suggestions'] = self.generate_improvement_suggestions(phase_scores)
        
        return result
    
    def visualize_score_comparison(self, scores, output_path=None):
        """可视化各阶段得分对比"""
        phases = list(scores['phase_scores'].keys())
        scores_values = [scores['phase_scores'][p]['score'] for p in phases]
        weights = [scores['phase_scores'][p]['weight'] for p in phases]
        
        # 转换为中文阶段名称
        phase_names = [self.phase_names.get(p, p) for p in phases]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 柱状图
        bars = ax.bar(phase_names, scores_values, color=['#3498db', '#2ecc71', '#e74c3c'])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}',
                   ha='center', va='bottom')
        
        # 添加总分
        ax.axhline(y=scores['overall_score'], color='r', linestyle='--', alpha=0.7)
        ax.text(len(phases) - 0.5, scores['overall_score'] + 0.02, 
               f"总分: {scores['overall_score']:.2f}", 
               ha='center', va='bottom', color='red', fontweight='bold')
        
        # 设置图表
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('得分')
        ax.set_title('跳远各阶段评分对比')
        
        # 添加权重标签
        for i, (p, w) in enumerate(zip(phase_names, weights)):
            ax.text(i, 0.05, f'权重: {w:.2f}', ha='center', va='bottom', color='gray')
        
        # 添加整体评价
        plt.figtext(0.5, 0.01, scores['overall_evaluation'], ha='center', fontsize=12, 
                   bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout()
        
        # 保存或显示
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def save_score_report(self, scores, output_path):
        """保存评分报告为JSON文件"""
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存为JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(scores, f, indent=4, ensure_ascii=False)
                
            print(f"\n评分报告已保存到: {output_path}")
            
            # 生成可视化结果
            vis_path = output_path.replace('.json', '_chart.png')
            self.visualize_score_comparison(scores, vis_path)
            
            return True
        except Exception as e:
            print(f"保存评分报告时出错: {e}")
            return False