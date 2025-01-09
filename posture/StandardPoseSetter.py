import cv2
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from Human.Human import Human
import os
import numpy as np

class StandardPoseSetter:
    def __init__(self, master):
        self.master = master
        self.master.title("标准姿态设定器")
        
        self.video_path = ""
        self.cap = None
        self.current_frame = None
        self.total_frames = 0  # 添加总帧数变量
        
        self.human = Human(config_path='config/config.ini')
        self.human.init_model()
        
        self.keypoints = {}
        
        # 按钮：加载视频
        self.load_button = tk.Button(master, text="加载视频", command=self.load_video)
        self.load_button.pack()
        
        # 视频显示区域
        self.video_label = tk.Label(master)
        self.video_label.pack()
        
        # 滑动条：选择帧
        self.frame_slider = tk.Scale(master, from_=0, to=0, orient=tk.HORIZONTAL, label="选择帧",
                                     command=self.on_slider_change, length=400)
        self.frame_slider.pack()
        self.frame_slider.config(state=tk.DISABLED)
        
        # 按钮：选择帧
        self.select_frame_button = tk.Button(master, text="进行姿态估计", command=self.select_frame, state=tk.DISABLED)
        self.select_frame_button.pack()
        
        # 按钮：保存关键点
        self.save_button = tk.Button(master, text="保存关键点", command=self.save_keypoints, state=tk.DISABLED)
        self.save_button.pack()
    
    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_slider.config(to=self.total_frames - 1)
            self.frame_slider.config(state=tk.NORMAL)
            self.select_frame_button.config(state=tk.NORMAL)
            self.show_frame(0)  # 显示第一帧
    
    def show_frame(self, frame_number):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
    
    def on_slider_change(self, val):
        frame_number = int(val)
        self.show_frame(frame_number)
    
    def select_frame(self):
        if self.current_frame is not None:
            bboxes = self.human.get_bbox_one_image(self.current_frame)
            if bboxes is None or len(bboxes) == 0:
                messagebox.showerror("错误", "未检测到任何目标。")
                return
            
            # 选择最大的检测框
            largest_bbox = max(bboxes, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            
            keypoints = self.human.get_pose_one_image(self.current_frame, [largest_bbox])
            if keypoints:
                self.keypoints = keypoints[0]['keypoints']  # 只保存第一个目标的关键点
                self.show_visualization(keypoints)
                messagebox.showinfo("成功", "姿态估计完成，可以保存关键点数据。")
                self.save_button.config(state=tk.NORMAL)
            else:
                messagebox.showerror("错误", "姿态估计失败。")
        else:
            messagebox.showerror("错误", "没有可选择的帧。")
    
    def show_visualization(self, keypoints):
        # 创建新窗口显示姿态估计结果
        vis_window = tk.Toplevel(self.master)
        vis_window.title("姿态估计结果")
    
        # 绘制关键点和连线
        vis_image = self.human.draw_pose(self.current_frame.copy(), keypoints)
        cv2image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
    
        label = tk.Label(vis_window, image=imgtk)
        label.imgtk = imgtk  # 需要保持引用
        label.pack()
    
        # 按钮：保存可视化结果
        save_vis_button = tk.Button(vis_window, text="保存可视化结果", 
                                    command=lambda: self.save_visualization(vis_image))
        save_vis_button.pack()
    
    def save_visualization(self, image):
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG Files", "*.jpg"), ("All Files", "*.*")])
        if save_path:
            cv2.imwrite(save_path, image)
            messagebox.showinfo("成功", f"可视化结果已保存到 {save_path}")
    
    def save_keypoints(self):
        if self.keypoints:
            # 添加下拉菜单选择姿态类型
            pose_type_window = tk.Toplevel(self.master)
            pose_type_window.title("选择姿态类型")
            pose_type_window.geometry("200x150")
            
            pose_type_var = tk.StringVar(pose_type_window)
            pose_type_var.set("takeoff")  # 默认值
            
            # 姿态类型选项
            pose_types = {
                "takeoff": "标准起跳姿势",
                "flight": "标准空中姿势",
                "landing": "标准落地姿势"
            }
            
            # 创建选择框
            option_menu = tk.OptionMenu(pose_type_window, pose_type_var, *pose_types.keys())
            option_menu.pack(pady=10)
            
            def save_with_type():
                pose_type = pose_type_var.get()
                description = pose_types[pose_type]
                
                # 检查keypoints类型并转换
                keypoints_data = self.keypoints
                if isinstance(keypoints_data, np.ndarray):
                    keypoints_data = keypoints_data.tolist()
                
                # 创建标准格式的数据
                standard_pose = {
                    "version": "1.0",
                    "pose_type": pose_type,
                    "keypoints": {
                        "coordinates": keypoints_data,  # 直接使用转换后的数据
                        "scores": [0.9] * len(keypoints_data)  # 设置默认置信度
                    },
                    "description": description
                }
                
                # 默认保存路径
                default_paths = {
                    "takeoff": "public/data/standard_poses/takeoff/root.json",
                    "flight": "public/data/standard_poses/flight/hip.json",
                    "landing": "public/data/standard_poses/landing/adb.json"
                }
                
                default_path = default_paths[pose_type]
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    initialfile=os.path.basename(default_path),
                    initialdir=os.path.dirname(default_path),
                    filetypes=[("JSON Files", "*.json")]
                )
                
                if save_path:
                    # 确保目录存在
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    # 保存JSON文件
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(standard_pose, f, indent=4, ensure_ascii=False)
                    
                    # 同时保存可视化结果
                    vis_path = save_path.replace('.json', '_visual.jpg')
                    cv2.imwrite(vis_path, self.current_frame)
                    
                    messagebox.showinfo("成功", f"标准姿态数据已保存到:\n{save_path}\n可视化结果已保存到:\n{vis_path}")
                    pose_type_window.destroy()
            
            # 添加保存按钮
            save_button = tk.Button(pose_type_window, text="保存", command=save_with_type)
            save_button.pack(pady=10)
            
        else:
            messagebox.showerror("错误", "没有关键点数据可保存。")

if __name__ == "__main__":
    root = tk.Tk()
    app = StandardPoseSetter(root)
    root.mainloop() 