import json
import numpy as np
import os
import datetime

class KeyScoreCalculator:
    def __init__(self):
        self.pose_types = ['takeoff', 'flight', 'landing']
        self.weight_files = {
            'takeoff': 'public/data/standard_poses/takeoff/weight_takeoff.json',
            'flight': 'public/data/standard_poses/flight/weight_flight.json',
            'landing': 'public/data/standard_poses/landing/weight_landing.json'
        }
        # MMPose 17个关键点的名称映射
        self.keypoint_names = {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
            5: "left_shoulder",
            6: "right_shoulder",
            7: "left_elbow",
            8: "right_elbow",
            9: "left_wrist",
            10: "right_wrist",
            11: "left_hip",
            12: "right_hip",
            13: "left_knee",
            14: "right_knee",
            15: "left_ankle",
            16: "right_ankle"
        }
        # 添加角度名称映射
        self.angle_descriptions = {
            'takeoff': {
                (12, 14, 16): "右髋-右膝-右踝",
                (13, 12): "左膝-右髋与水平线夹角",
                (11, 13, 15): "左髋-左膝-左踝"
            },
            'flight': {
                (10, 6): "右手腕-右肩与垂直线夹角",
                (9, 5): "左手腕-左肩与垂直线夹角",
                (6, 12, 14): "右肩-右髋-右膝",
                (5, 11, 13): "左肩-左髋-左膝"
            },
            'landing': {
                (6, 12, 14): "右肩-右髋-右膝",
                (5, 11, 13): "左肩-左髋-左膝",
                (14, 12, 16): "右膝-右髋-右踝",
                (13, 11, 15): "左膝-左髋-左踝"
            }
        }

    def calculate_angle(self, point1, point2, point3):
        """计算三个点形成的角
        Args:
            point1: 第一个点的坐标
            point2: 中间点（角的顶点）的坐标
            point3: 第三个点的坐标
        """
        try:
            vector1 = point1 - point2  # 从顶点指向第一个点的向量
            vector2 = point3 - point2  # 从顶点指向第三个点的向量
            
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
            return angle_degrees
            
        except Exception as e:
            print(f"计算角度时出错: {str(e)}")
            return 0.0

    def calculate_key_angles(self, keypoints):
        """计算关键点之间的角度"""
        angles = {}
        keypoints = np.array(keypoints)
        
        for i in range(len(keypoints)):
            for j in range(len(keypoints)):
                for k in range(len(keypoints)):
                    key = f"{i},{j},{k}"
                    angles[key] = self.calculate_angle(
                        keypoints[i][:2],
                        keypoints[j][:2],
                        keypoints[k][:2]
                    )
        
        return angles

    def get_score_from_ranges(self, angle, score_ranges):
        """根据角度范围获取分数"""
        # 首先检查是否正好在某个范围的边界上
        for range_info in score_ranges:
            start, end = range_info['range']
            min_score = range_info.get('min_score', 0)
            max_score = range_info['score']
            
            # 如果角度好在边界上，选择得分更高的范围
            if angle == start or angle == end:
                next_range = None
                # 找到相邻的范围
                for r in score_ranges:
                    if r['range'][0] == end or r['range'][1] == start:
                        next_range = r
                        break
                
                # 比较两个范围的分数，选择更高的
                if next_range:
                    score1 = max_score
                    score2 = next_range['score']
                    if score1 >= score2:
                        return score1
                    else:
                        return score2
            
            # 正常范围判断
            if start < angle < end:
                if min_score == max_score:
                    return max_score
                
                # 计算在范围内的比例
                ratio = (angle - start) / (end - start)
                
                # 对于递减的分数段，需要反转比例
                if max_score < min_score:
                    ratio = 1 - ratio
                    score = max_score + ratio * (min_score - max_score)
                else:
                    score = min_score + ratio * (max_score - min_score)
                    
                print(f"DEBUG: 范围[{start}°-{end}°], 实际角度={angle:.1f}°, "
                      f"比例={ratio:.2f}, 分数范围[{min_score}-{max_score}], "
                      f"计算得分={score:.1f}")
                
                return score
        
        # 如果不在任何范围内，找到最接近的范围
        closest_range = min(score_ranges, 
                           key=lambda x: min(abs(angle - x['range'][0]), 
                                           abs(angle - x['range'][1])))
        start, end = closest_range['range']
        
        # 如果角度超出范围，返回该范围的最小或最大分数
        if angle < start:
            return closest_range.get('min_score', 0)
        else:
            return closest_range.get('min_score', 0)

    def calculate_angle_with_horizontal(self, point1, point2):
        """计算向量与水平线的夹角"""
        try:
            # 创建一个水平向量
            horizontal = np.array([1, 0])
            # 计算两点之间的向量
            vector = point1 - point2
            
            # 计算点积
            dot_product = np.dot(vector, horizontal)
            # 计算向量的模
            norm = np.linalg.norm(vector)
            
            # 计算角度（弧度）
            cos_angle = dot_product / (norm * np.linalg.norm(horizontal))
            # 处理数值误差
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # 转换为角度
            angle_degrees = np.degrees(angle)
            
            # 根据向量的y坐标判断角度的方向
            if vector[1] < 0:  # 如果y是负数，说明向量朝上
                angle_degrees = 360 - angle_degrees
                
            return angle_degrees
            
        except Exception as e:
            print(f"计算水平夹角时出错: {str(e)}")
            return 0.0

    def generate_reference_point(self, base_point, angle_type='horizontal', direction='right', distance=100):
        """生成参考点
        Args:
            base_point: 基准点坐标 [x, y]
            angle_type: 'horizontal' 或 'vertical'
            direction: 'right'/'left' (对于水平线) 或 'up'/'down' (对于垂直线)
            distance: 生成点的距离
        Returns:
            reference_point: 生成的参考点坐标 [x, y]
        """
        if angle_type == 'horizontal':
            # 水平线
            if direction == 'right':
                return np.array([base_point[0] + distance, base_point[1]])
            else:  # left
                return np.array([base_point[0] - distance, base_point[1]])
        else:  # vertical
            # 垂直线
            if direction == 'up':
                return np.array([base_point[0], base_point[1] - distance])
            else:  # down
                return np.array([base_point[0], base_point[1] + distance])

    def calculate_pose_score(self, pose_type, matched_pose_file):
        """计算特定姿态的得分"""
        try:
            # 加载权重文件
            weight_file = self.weight_files.get(pose_type)
            print(f"\n=== 开始计算 {pose_type} 姿态得分 ===")
            print(f"权重文件路径: {weight_file}")
            print(f"匹配姿态文件路径: {matched_pose_file}")

            if not weight_file or not os.path.exists(weight_file):
                print(f"未找到权重文件: {weight_file}")
                return 0.0

            # 加载标准姿态文件
            standard_pose_path = os.path.join('public/data/standard_poses', pose_type, 'root.json')
            print(f"标准姿态文件路径: {standard_pose_path}")

            # 加载标准姿态坐标
            with open(standard_pose_path, 'r') as f:
                standard_data = json.load(f)
                standard_coordinates = np.array(standard_data['keypoints']['coordinates'])

            with open(weight_file, 'r') as f:
                weights = json.load(f)
                

            # 加载匹配到的姿态文件
            with open(matched_pose_file, 'r') as f:
                matched_data = json.load(f)
                print(f"找到 {len(matched_data.get('matches', []))} 个匹配结果")

            # 获取最佳匹配的关键点
            best_match = matched_data['matches'][0]
            matched_keypoints = np.array(best_match['keypoints'])

            print("\n开始计算各关键点角度得分:")
            total_score = 0.0
            total_weight = 0.0

            # 创建评分结果字典
            score_result = {
                "pose_type": pose_type,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "overall_score": 0.0,
                "angles_evaluation": [],
                "suggestions": [],
                "details": {
                    "matched_pose_file": matched_pose_file,
                    "weight_file": weight_file,
                    "standard_pose_file": standard_pose_path
                }
            }

            for weight_item in weights:
                angle_result = {
                    "angle_name": f"V{len(score_result['angles_evaluation']) + 1}",
                    "key_points": [self.keypoint_names[i] for i in weight_item['key_angle']],
                    "angle_description": self.get_angle_description(pose_type, tuple(weight_item['key_angle'])),
                    "weight": weight_item['weight'],
                    "recommended_angle": weight_item['angle'],
                    "actual_angle": 0.0,
                    "score": 0.0,
                    "score_ranges": weight_item['score_ranges'],
                    "evaluation": ""
                }

                key_angle = weight_item['key_angle']
                weight = weight_item['weight']
                recommended_angle = weight_item['angle']
                score_ranges = weight_item['score_ranges']
                angle_type = weight_item.get('angle_type', 'normal')
                vertex_index = weight_item.get('vertex_index', 1)
                direction = weight_item.get('direction', 'right')

                point_names = [self.keypoint_names[i] for i in key_angle]

                # 打印角度类型和关键点信息
                print(f"\n关键点 {key_angle} ({' -> '.join(point_names)}):")

                # 获取匹配姿态的点坐标
                if angle_type in ['horizontal', 'vertical']:
                    # 对于水平/垂直角度，生成参考点
                    p1 = matched_keypoints[key_angle[0]][:2]
                    p2 = matched_keypoints[key_angle[1]][:2]  # 顶点
                    p3 = self.generate_reference_point(p2, angle_type, direction)
                    
                    print(f"计算{angle_type}度 (方向: {direction})")
                    print("匹配姿态坐标:")
                    print(f"  第一点: {p1} ({point_names[0]})")
                    print(f"  顶点(参考点): {p2} ({point_names[1]})")
                    print(f"  生成的{angle_type}点: {p3} ({direction}方向)")
                else:
                    # 普通三点角度
                    points = [matched_keypoints[i][:2] for i in key_angle]
                    vertex = points.pop(vertex_index)
                    p1, p2, p3 = points[0], vertex, points[1]
                    
                    print("计算三点角度")
                    print("匹配姿态坐标:")
                    print(f"  第一点: {p1} ({point_names[0]})")
                    print(f"  顶点: {p2} ({point_names[vertex_index]})")
                    print(f"  第三点: {p3} ({point_names[-1]})")

                # 计算角度
                actual_angle = self.calculate_angle(p1, p2, p3)

                # 根据角度范围获取分数
                angle_score = self.get_score_from_ranges(actual_angle, score_ranges)
                weighted_score = (angle_score / 100.0) * weight

                # 更新角度结果
                angle_result["actual_angle"] = actual_angle
                angle_result["score"] = angle_score

                # 生成评价建议
                if angle_score >= 80:
                    angle_result["evaluation"] = "优秀"
                elif angle_score >= 60:
                    angle_result["evaluation"] = "良好"
                else:
                    angle_result["evaluation"] = "需要改进"
                    
                    # 根据不同动作类型生成具体建议
                    if pose_type == "takeoff":
                        if "knee" in angle_result["key_points"][1].lower():
                            score_result["suggestions"].append(f"起跳时膝关节弯曲角度不足，建议加强腿部力量训练")
                    elif pose_type == "flight":
                        if "shoulder" in angle_result["key_points"][0].lower():
                            score_result["suggestions"].append(f"空中手臂后摆不充分，建议加强手臂摆动幅度")
                    elif pose_type == "landing":
                        if "hip" in angle_result["key_points"][1].lower():
                            score_result["suggestions"].append(f"落地时收腹不够，建议加强核心力量训练")

                score_result["angles_evaluation"].append(angle_result)

                total_score += weighted_score
                total_weight += weight

                print(f"计算结果:")
                print(f"  实际角度: {actual_angle:.1f}°")
                print(f"  分数段得分: {angle_score*100:.0f}")
                print(f"  权重: {weight:.2f}")
                print(f"  推荐角度: {recommended_angle}°")
                print(f"  分数段范围:")
                for range_info in score_ranges:
                    min_score = range_info.get('min_score', 0)
                    max_score = range_info['score']
                    start, end = range_info['range']
                    print(f"    {start}°-{end}° : {min_score}-{max_score}分")

            # 计算总分
            final_score = total_score / total_weight if total_weight > 0 else 0
            final_score = final_score * 100
            score_result["overall_score"] = final_score

            # 生成总体评价
            if final_score >= 90:
                score_result["overall_evaluation"] = "动作完成度优秀"
            elif final_score >= 80:
                score_result["overall_evaluation"] = "动作完成度良好"
            elif final_score >= 60:
                score_result["overall_evaluation"] = "动作基本合格，有提升空间"
            else:
                score_result["overall_evaluation"] = "动作需要改进"

            # 保存评分结果
            result_file = os.path.join(os.path.dirname(matched_pose_file), f"{pose_type}_score_result.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(score_result, f, ensure_ascii=False, indent=4)

            print(f"\n{pose_type} 姿态最终得分: {final_score:.2f}")
            print("=== 计算完成 ===\n")
            
            return final_score

        except Exception as e:
            print(f"计算姿态得分时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0

    def calculate_all_scores(self, result_folder):
        """计算所有姿态的得分"""
        scores = {}
        
        for pose_type in self.pose_types:
            pose_dir = os.path.join(result_folder, pose_type)
            matched_pose_file = os.path.join(pose_dir, 'pose_matches.json')
            
            if os.path.exists(matched_pose_file):
                score = self.calculate_pose_score(pose_type, matched_pose_file)
                scores[f'{pose_type}_key_score'] = score
                print(f"{pose_type} 关键点得分: {score:.2f}")
            else:
                print(f"未找到匹配文件: {matched_pose_file}")
                scores[f'{pose_type}_key_score'] = 0.0

        return scores

    def generate_jump_score(self, jump_dir):
        """生成跳远总评分"""
        try:
            # 读取三个阶段的评分结果
            phases = ['takeoff', 'flight', 'landing']
            phase_scores = {}
            phase_weights = {
                'takeoff': 0.3,    # 起跳占30%
                'flight': 0.4,     # 空中占40%
                'landing': 0.3     # 落地占30%
            }
            
            total_score = 0.0
            jump_result = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "overall_score": 0.0,
                "phase_scores": {},
                "overall_evaluation": "",
                "phase_evaluations": [],
                "suggestions": [],
                "details": {}
            }

            # 使用集合来存储建议，自动去重
            unique_suggestions = set()

            # 收集各阶段得分
            for phase in phases:
                score_file = os.path.join(jump_dir, phase, f"{phase}_score_result.json")
                if os.path.exists(score_file):
                    with open(score_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        phase_scores[phase] = result["overall_score"]
                        
                        # 添加阶段评分详情
                        jump_result["phase_scores"][phase] = {
                            "score": result["overall_score"],
                            "weight": phase_weights[phase],
                            "weighted_score": result["overall_score"] * phase_weights[phase]
                        }
                        
                        # 收集各阶段的评价和建议
                        jump_result["phase_evaluations"].append({
                            "phase": phase,
                            "evaluation": result["overall_evaluation"],
                            "angles_evaluation": result["angles_evaluation"]
                        })
                        
                        # 收集改进建议（使用集合去重）
                        unique_suggestions.update(result["suggestions"])
                        
                        # 计算加权总分
                        total_score += result["overall_score"] * phase_weights[phase]
                else:
                    print(f"警告: 未找到{phase}阶段��评分文件")
                    jump_result["phase_scores"][phase] = {
                        "score": 0.0,
                        "weight": phase_weights[phase],
                        "weighted_score": 0.0
                    }

            # 将去重后的建议转换为列表
            jump_result["suggestions"] = list(unique_suggestions)

            # 设置总分和总体评价
            jump_result["overall_score"] = total_score
            
            if total_score >= 90:
                jump_result["overall_evaluation"] = "跳远动作完成度优秀"
            elif total_score >= 80:
                jump_result["overall_evaluation"] = "跳远动作完成度良好"
            elif total_score >= 60:
                jump_result["overall_evaluation"] = "跳远动作基本合格，有提升空间"
            else:
                jump_result["overall_evaluation"] = "跳远动作需要改进"

            # 添加详细信息
            jump_result["details"] = {
                "jump_directory": jump_dir,
                "phase_files": {
                    phase: os.path.join(jump_dir, phase, f"{phase}_score_result.json")
                    for phase in phases
                }
            }

            # 保存总评分结果
            result_file = os.path.join(jump_dir, "jump_score_result.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(jump_result, f, ensure_ascii=False, indent=4)

            print(f"\n=== 跳远总评分完成 ===")
            print(f"总分: {total_score:.2f}")
            for phase in phases:
                print(f"{phase}得分: {phase_scores.get(phase, 0.0):.2f}")
            print(f"评分结果已保存至: {result_file}\n")

        except Exception as e:
            print(f"生成跳远总评分时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_angle_description(self, pose_type, key_angle):
        """获取角度的中文描述"""
        if pose_type in self.angle_descriptions:
            # 确保键的格式正确
            key_angle = tuple(key_angle)  # 转换为元组
            
            # 调试输出
            print(f"DEBUG: 查找角度描述 - 阶段: {pose_type}, 关键点: {key_angle}")
            print(f"DEBUG: 可用的角度描述: {self.angle_descriptions[pose_type].keys()}")
            
            description = self.angle_descriptions[pose_type].get(key_angle)
            if description is None:
                # 如果没找到描述，生成一个基本描述
                points = [self.keypoint_names[i] for i in key_angle]
                if len(points) == 2:
                    return f"{points[0]}-{points[1]}与参考线夹角"
                else:
                    return f"{points[0]}-{points[1]}-{points[2]}角度"
            return description
        
        return "未命名角度" 