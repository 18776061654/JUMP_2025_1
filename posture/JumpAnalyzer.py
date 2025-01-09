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
                            # 记录关键点和分数
                            frame_info = {
                                'frame_id': self.current_frame,
                                'keypoints': keypoints.tolist(),
                                'scores': keypoint_scores.tolist(),
                                'bbox': bbox.tolist()
                            }
                            self.trajectory.append(frame_info)
                            
                            # 在帧上绘制结果
                            self.draw_results(frame, bbox, keypoints, keypoint_scores)
                
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
        
        cap.release()

    def draw_results(self, frame: np.ndarray, bbox: np.ndarray, keypoints: np.ndarray, keypoint_scores: np.ndarray):
        """绘制检测框、骨架和关键点"""
        try:
            # 确保bbox是正确的格式
            bbox = np.array(bbox).astype(int)  # 转换为整数坐标
            if len(bbox) == 4:  # 确保有4个坐标点
                # 绘制检测框
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {self.target_id}", 
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
                'trajectory': self.trajectory
            }
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存为JSON文件
            with open(output_path, 'w') as f:
                json.dump(result_data, f, indent=4)
                
            print(f"\n分析结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"保存结果时出错: {str(e)}")

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
            return np.array([0, 0, 100, 100])  # 返回一��默认的边界框

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
                    
                    # 在帧上绘制结果
                    frame_with_pose = frame.copy()
                    self.draw_results(frame_with_pose, bbox, keypoints, keypoint_scores)
                    return frame_with_pose
            
            return frame
            
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
            return frame

def main():
    """主函数"""
    # 创建 Tkinter 根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 打开文件选择对话框
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

    print(f"已选择视频文件: {video_path}")

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"\n视频信息:")
    print(f"总帧数: {total_frames}")
    print(f"帧率: {fps} fps")
    print(f"预计时长: {total_frames/fps:.1f} 秒")

    # 读取第一帧
    ret, first_frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        cap.release()
        return

    print("\n请在视频第一帧中选择起跳预备区域（点击两个点以��定矩形区域）...")
    # 让用户选择预备区域
    roi = select_roi(first_frame.copy())
    if roi is None:
        print("未选择预备区域")
        cap.release()
        return

    print(f"已设置预备区域: {roi}")

    # 创建分析器
    analyzer = JumpAnalyzer()
    analyzer.set_start_zone(*roi)

    # 创建结果目录
    result_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result')
    os.makedirs(result_dir, exist_ok=True)

    # 设置输出文件路径
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(result_dir, f"{video_name}_analyzed.mp4")
    output_json_path = os.path.join(result_dir, f"{video_name}_analysis.json")

    # 创建输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 重置视频到开始
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("\n开始处理视频...")
    frame_count = 0
    tracking_started = False
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # 计算进度
            progress = (frame_count / total_frames) * 100
            
            # 处理当前帧
            processed_frame, frame_info = analyzer.process_frame(frame, frame_count - 1)

            # 显示进度信息
            if frame_info:
                if not tracking_started:
                    print("\n检测到目标进入预备区域，开始追踪和姿态估计...")
                    tracking_started = True
                status = "正在追踪和分析"
            else:
                status = "等待目标进入预备区域" if not tracking_started else "追踪中"

            print(f"\r处理进度: {frame_count}/{total_frames} ({progress:.1f}%) - {status}", end="")

            # 显示处理后的帧
            cv2.imshow("Analysis", processed_frame)
            out.write(processed_frame)

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户中断处理")
                break

    finally:
        # 保存结果
        if analyzer.trajectory:
            analyzer.save_results(output_json_path)
            print(f"\n\n处理完成!")
            print(f"已保存:")
            print(f"- 处理后的视频: {output_video_path}")
            print(f"- 分析数据: {output_json_path}")
            print(f"\n分析统计:")
            print(f"- 总帧数: {total_frames}")
            print(f"- 分析数: {len(analyzer.trajectory)}")
            print(f"- 分析成功率: {(len(analyzer.trajectory)/total_frames)*100:.1f}%")
        else:
            print("\n\n处理完成，但未能成功追踪和分析目标")

        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def select_roi(frame):
    """
    让用户在视频第一帧选择两个点来创建预备区域
    :param frame: 视频第一帧
    :return: (x, y, width, height) 或 None
    """
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # 绘制点
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            if len(points) == 2:
                # 绘制矩形
                x1, y1 = points[0]
                x2, y2 = points[1]
                cv2.rectangle(frame, points[0], points[1], (255, 0, 0), 2)
            cv2.imshow("Select ROI", frame)

    # 创建窗口并设置鼠标回调
    cv2.imshow("Select ROI", frame)
    cv2.setMouseCallback("Select ROI", mouse_callback)

    while len(points) < 2:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
            cv2.destroyWindow("Select ROI")
            return None
    
    cv2.destroyWindow("Select ROI")
    
    # 计算矩形坐标
    x1, y1 = points[0]
    x2, y2 = points[1]
    x = min(x1, x2)
    y = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    
    return (x, y, width, height)

if __name__ == "__main__":
    main() 