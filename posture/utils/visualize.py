import utils.font_util as font_util
import numpy as np
import cv2
from .angle import read_points_from_file
import os

def back_to_origin(points,image_shape):
    origin_points = []
    height,width,_ = image_shape
    for point in points:
        origin_points.append([point[0] * width,point[1] * height])
    return origin_points

def construct_point(landmarks):
    """按照挺身式跳远重构关键点

    Args:
        landmarks (_type_): _description_

    Returns:
        points: 重构后的关键点
    """
    construct_num = [11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]
    points = []
    i = 0
    
    #取鼻子,肩中点和髋中点
    nose = [(landmarks[0].x),(landmarks[0].y)]
    points.append(nose)
    
    shoulder_mid = [((landmarks[11].x + landmarks[12].x)/2),((landmarks[11].y + landmarks[12].y)/2)]
    points.append(shoulder_mid)
    
    hip_mid = [((landmarks[23].x + landmarks[24].x)/2),((landmarks[23].y + landmarks[24].y)/2)]
    points.append(hip_mid)
    
    
    while i < len(construct_num):
        points.append([landmarks[construct_num[i]].x , landmarks[construct_num[i]].y ])
        i = i + 1 
    
    return points

def plot_posture(image, frame_id=0, fps=0.,points=[]):
    
    im = np.ascontiguousarray(np.copy(image))
    text_scale = 2
    
    # 打中文  
    text = ''
    font_size = int(min(im.shape[1], image.shape[0]) * 0.035)
    im = font_util.cv2_chinese_text(im, text, (5, 40), (255, 0, 0), font_size)  
            
    cv2.putText(im, 'frame: %d fps: %.2f ' % (frame_id, fps),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
    
    if points != []:
        im = plot_construct_point(im,points,color=(0, 255, 0))

    return im


def plot_construct_point(image,points,color=(0, 0, 255)):
    """将点数组画成人体结构

    Args:
        image (_type_): _description_
        points (_type_): _description_
        color (tuple, optional): _description_. Defaults to (0, 0, 255).

    Returns:
        image: _description_
    """
    points = [(int(x), int(y)) for x, y in points]

    # 画头,脖子，髋
    cv2.line(image, points[0], points[1], color, 2, cv2.LINE_AA)
    cv2.line(image, points[1], points[2], color, 2, cv2.LINE_AA)
    
    # 画骨架
    skeleton_points = [points[3],points[4],points[10],points[9]]
    cv2.polylines(image, [np.array(skeleton_points)] , isClosed=True, color = color, thickness=2)
    
    # 画手臂
    cv2.line(image, points[3], points[5], color, 2, cv2.LINE_AA)
    cv2.line(image, points[5], points[7], color, 2, cv2.LINE_AA)
    cv2.line(image, points[4], points[6], color, 2, cv2.LINE_AA)
    cv2.line(image, points[6], points[8], color, 2, cv2.LINE_AA)
    
    # 画腿
    cv2.line(image, points[9], points[11], color, 2, cv2.LINE_AA)
    cv2.line(image, points[11], points[13], color, 2, cv2.LINE_AA)
    cv2.line(image, points[10], points[12], color, 2, cv2.LINE_AA)
    cv2.line(image, points[12], points[14], color, 2, cv2.LINE_AA)
    
    # 画脚
    left_foot_points = [points[13],points[15],points[17]]
    right_foot_points = [points[14],points[16],points[18]]
    cv2.polylines(image, [np.array(left_foot_points)] , isClosed=True, color = color, thickness=2)
    cv2.polylines(image, [np.array(right_foot_points)] , isClosed=True, color = color, thickness=2)
    
    # 画点
    for point in points:
        cv2.circle(image, point, 1, (255, 0, 0), 2)
    
           
    return image


def create_canvas():
    """创建网格画布,大小为1280*720,暂时不支持更改

    Returns:
        canvas: image type
    """
    # 创建画布
    width, height = 1280, 720
    canvas = cv2.cvtColor(np.ones((height, width, 3), dtype=np.uint8) * 255, cv2.COLOR_BGR2RGB)  # 创建白色背景画布

    # 绘制网格线和刻度
    grid_color = (200, 200, 200)  # 网格线颜色
    tick_color = (0, 0, 0)  # 刻度颜色
    grid_size = 100  # 网格大小

    # 绘制网格线
    for x in range(grid_size, width, grid_size):
        cv2.line(canvas, (x, 0), (x, height), grid_color, 1)
    for y in range(grid_size, height, grid_size):
        cv2.line(canvas, (0, y), (width, y), grid_color, 1)

    # 绘制刻度和标签
    for i in range(1, 13):
        x = i * grid_size
        y = i * grid_size

        # x轴刻度
        cv2.line(canvas, (x, 0), (x, 10), tick_color, 2)
        cv2.putText(canvas, str(i * 100), (x - 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tick_color, 1, cv2.LINE_AA)

        # y轴刻度
        cv2.line(canvas, (10, y), (0, y), tick_color, 2)
        cv2.putText(canvas, str(i * 100), (20, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tick_color, 1, cv2.LINE_AA)

    # 绘制坐标轴
    axis_color = (0, 0, 0)  # 坐标轴颜色

    # x轴
    cv2.line(canvas, (0, 0), (width, 0), axis_color, 2)
    cv2.putText(canvas, 'x', (width - 20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, axis_color, 1, cv2.LINE_AA)

    # y轴
    cv2.line(canvas, (0, 0), (0, height), axis_color, 2)
    cv2.putText(canvas, 'y', (20, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, axis_color, 1, cv2.LINE_AA)
     
    return canvas

def posture_visualize(test_points,posture_type,save_folder):
    # 标准姿态和待测姿态可视化对比
    filename = posture_type + '.txt'
    
    canvas = create_canvas()   
        
    standard_points = read_points_from_file(filename)
    test_points = back_to_origin(test_points,canvas.shape)
    standard_points = back_to_origin(standard_points,canvas.shape)

    # 假设腰部的点是点列表中的第一个点
    waist_test_point = test_points[2]
    waist_standard_point = standard_points[2]

    # 计算平移向量
    translation_vector = (waist_standard_point[0] - waist_test_point[0], waist_standard_point[1] - waist_test_point[1])

    # 应用平移
    translated_test_points = [(x + translation_vector[0], y + translation_vector[1]) for x, y in test_points]

    # 绘制点并保存
    canvas = plot_construct_point(canvas, translated_test_points, color=(0, 255, 0))
    canvas = plot_construct_point(canvas, standard_points)
    cv2.imwrite(os.path.join(save_folder, 'compare_' + posture_type + '.jpg'), canvas)
    # base_point = [standard_points[1],standard_points[2]]
    # base_test_point = [test_points[1],test_points[2]]
    # T = calculate_affine_transform(base_test_point,base_point)
    
    # base_test_point = apply_affine_transform(base_test_point,T)
    # affine_test_points = apply_affine_transform(test_points,T)
    
    # canvas = plot_construct_point(canvas,affine_test_points,color=(0, 255, 0))
    # canvas = plot_construct_point(canvas,standard_points)
    
    # cv2.imwrite(os.path.join(save_folder, 'compare_' + posture_type + '.jpg'),canvas)      
    

def calculate_affine_transform(points1, points2):
    # 将点数组转换为numpy数组
    points1 = np.array(points1)
    points2 = np.array(points2)
    
    # 确保点数组包含两个点
    assert points1.shape == (2, 2)
    assert points2.shape == (2, 2)
    
    # 构造齐次坐标矩阵
    P = np.vstack((points1.T, np.ones(2)))
    Q = np.vstack((points2.T, np.ones(2)))
    
    # 计算仿射变换矩阵T
    T, residuals, rank, s = np.linalg.lstsq(P.T, Q.T, rcond=None)
    
    return T.T

def apply_affine_transform(points1, T):
    # 将点数组转换为numpy数组
    points1 = np.array(points1)
    
    # 确保点数组的形状正确
    assert points1.shape[1] == 2
    
    # 构造齐次坐标矩阵
    P = np.hstack((points1, np.ones((points1.shape[0], 1))))
    
    # 应用仿射变换
    transformed_points = np.dot(T, P.T).T
    transformed_points = transformed_points[:, :2]
    
    return transformed_points
