from PySide6.QtCore import QThread, Signal
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from speed.human_with_tracking import HumanWithTracking  # 使用HumanWithTracking类

class Speed_VideoProcessor(QThread):
    processingFinished = Signal(str)
    error = Signal(str)
    
    def __init__(self, video_path, batch_size=4):
        super().__init__()
        self.video_path = video_path
        self.batch_size = batch_size  # 设置批量大小
        self.cap = None
        self.frame_speeds = []
        self.frame_id = 0
        self.video_writer = None
        self.mean_vector = 0
        self.human_tracker = HumanWithTracking(config_path='config/config.ini')

        try:
            self.human_tracker.init_model()
        except Exception as e:
            self.error.emit(f"模型初始化失败: {str(e)}")
    
    def run(self):
        if not self._initialize_video():
            return
        
        save_path = self._initialize_video_writer()
        if not save_path:
            return
        
        try:
            self._process_video_frames()
        except Exception as e:
            self.error.emit(f"视频处理时出错: {str(e)}")
        finally:
            self._release_resources()
        
        self._save_speed_plot(save_path)
        self._save_mean_vector(save_path)
        self.processingFinished.emit(self.video_path)

    def _initialize_video(self):
        """初始化视频资源，确保视频可以正常打开"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.error.emit("无法打开视频文件")
            return False
        
        self.CameraWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.CameraHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        return True

    def _initialize_video_writer(self):
        """初始化视频写入器"""
        file = self.video_path.split("/")[-1]
        file_type = "." + file.split(".")[1]
        save_folder = os.path.dirname(self.video_path)  # 获取视频所在目录
        
        # 确保在speed目录下
        if not save_folder.endswith('speed'):
            save_folder = os.path.join(save_folder, 'speed')
        os.makedirs(save_folder, exist_ok=True)
        
        save_path = os.path.join(save_folder, "SpeedVideo_analyzed" + file_type)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if file_type == ".mp4" else cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(save_path, fourcc, self.fps, (self.CameraWidth, self.CameraHeight))
        
        if not self.video_writer.isOpened():
            self.error.emit("无法初始化视频写入器")
            return None
        return save_folder

    def _process_video_frames(self):
        """批量处理视频"""
        self.frame_id = 0
        batch = []
        total_speed = 0
        num_frames = 0

        with ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.frame_id += 1
                cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch.append(cur_frame)

                # 如果批量满了，进行批量处理
                if len(batch) == self.batch_size:
                    future = executor.submit(self._process_batch, batch[:])
                    batch = batch[-1:]  # 保留最后一帧，作为下一批次的第一帧

                    # 获取批处理结果
                    speeds = future.result()
                    for speed in speeds:
                        if speed is not None and 2 <= speed <= 10:
                            total_speed += speed
                            num_frames += 1
                            self.frame_speeds.append(speed)
                            print(f"Processed batch, Speed: {speed:.2f} m/s")

        # 计算平均速度
        if num_frames > 0:
            self.mean_vector = total_speed / num_frames
    
    def _process_batch(self, batch):
        """批量处理帧并计算速度"""
        speeds = self.human_tracker.process_batch(batch, self.fps)
        return speeds

    def _release_resources(self):
        """释放视频资源"""
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
    
    def _save_speed_plot(self, save_folder):
        """保存速度变化图"""
        vectors = self.frame_speeds
        plt.figure()
        sampled_vectors = vectors[::3]
        plt.plot(sampled_vectors, marker='o')
        plt.xlabel('Time')
        plt.ylabel('Speed')
        plt.title('Speed Over Frames')
        plt.savefig(os.path.join(save_folder, 'speed_plot.png'))
        plt.close()  # 添加这行以释放内存

    def _save_mean_vector(self, save_folder):
        """保存平均速度到文本文件"""
        speed_save_path = os.path.join(save_folder, "speed_result.json")
        import json
        speed_data = {
            "average_speed": round(self.mean_vector, 2),
            "frame_speeds": [round(speed, 2) for speed in self.frame_speeds],
            "total_frames": len(self.frame_speeds)
        }
        with open(speed_save_path, 'w') as speed_file:
            json.dump(speed_data, speed_file, indent=4)
