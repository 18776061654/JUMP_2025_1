import cv2
import ffmpeg
import threading
import time
import logging
import os
import traceback


# 相机控制部分
class CameraController:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.cap = None
        self.running = False

    def check_camera_available(self):
        """检查摄像头是否可用"""
        try:
            cap = cv2.VideoCapture(self.camera_id)
            if cap is None or not cap.isOpened():
                return False
            ret, frame = cap.read()
            cap.release()
            return ret and frame is not None
        except Exception as e:
            print(f"检查摄像头 {self.camera_id} 时出错: {str(e)}")
            return False

    def start_camera(self):
        """启动摄像头"""
        try:
            if not self.running:
                self.cap = cv2.VideoCapture(self.camera_id)
                if self.cap is None or not self.cap.isOpened():
                    print(f"无法打开摄像头 {self.camera_id}")
                    return False
                self.running = True
                return True
            return False
        except Exception as e:
            print(f"启动摄像头 {self.camera_id} 时出错: {str(e)}")
            return False

    def stop_camera(self):
        """停止摄像头"""
        if self.running and self.cap is not None:
            self.cap.release()
            self.running = False
            self.cap = None

    def get_frame(self):
        """获取一帧图像"""
        if self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def __del__(self):
        """析构函数，确保摄像头被正确释放"""
        self.stop_camera()


class CameraRecorder:
    def __init__(self):
        self.processes = {}  # 存储每个摄像头的进程
        self.running = False

    def start_recording(self, camera_id, output_file):
        """开始摄像头录制"""
        if camera_id in self.processes:
            print(f"摄像头 '{camera_id}' 已在录制中。")
            return False

        try:
            print(f"开始录制视频到: {output_file}")  # 添加日志
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 修改FFmpeg参数以确保视频正确保存
            process = (
                ffmpeg
                .input(camera_id, format='dshow', rtbufsize='2000M')
                .output(output_file, 
                       vcodec='mjpeg',          # 使用 MJPEG 编码器
                       acodec='none',           # 不录制音频
                       r=30,                    # 设置帧率为30
                       vsync='cfr',             # 使用恒定帧率
                       video_bitrate='10000k',  # 设置视频比特率
                       f='avi')                 # 强制指定输出格式
                .global_args('-y')              # 覆盖已存在的文件
                .overwrite_output()
                .run_async(pipe_stdin=True, quiet=False)  # 添加 quiet=False 显示FFmpeg输出
            )
            
            self.processes[camera_id] = process
            self.running = True
            print(f"摄像头 '{camera_id}' 的录制已开始...")
            return True
            
        except Exception as e:
            print(f"启动录制时出错: {str(e)}")
            traceback.print_exc()
            return False

    def stop_recording(self, camera_id):
        """停止摄像头录制"""
        try:
            if camera_id not in self.processes:
                print(f"摄像头 '{camera_id}' 没有在录制中。")
                return True

            try:
                process = self.processes[camera_id]
                print(f"正在停止摄像头 '{camera_id}' 的录制...")
                
                # 尝试优雅地停止进程
                try:
                    process.stdin.write(b'q\n')
                    process.stdin.flush()
                except Exception as e:
                    print(f"发送停止信号时出错: {str(e)}")

                # 等待进程结束
                try:
                    process.wait(timeout=5)
                    print(f"摄像头 '{camera_id}' 的录制已成功停止")
                except Exception as e:
                    print(f"等待进程结束时出错: {str(e)}")
                    process.kill()
                    print(f"已强制终止录制进程")

            finally:
                # 清理进程
                self.processes.pop(camera_id, None)
                if not self.processes:
                    self.running = False
                
            return True
            
        except Exception as e:
            print(f"停止录制时出错: {str(e)}")
            traceback.print_exc()
            return False

    def __del__(self):
        """确保在对象销毁时停止所有录制"""
        try:
            # 复制键列表，因为我们会在迭代时修改字典
            camera_ids = list(self.processes.keys())
            for camera_id in camera_ids:
                self.stop_recording(camera_id)
        except Exception as e:
            logging.error(f"清理录制进程时出错: {str(e)}")
