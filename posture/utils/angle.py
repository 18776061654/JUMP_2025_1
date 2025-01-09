"""
此文件用于计算三个姿势下的角度加权距离
calculate_angle用于做最基础的夹角运算
calculate_take_off用于做起跳加权距离计算

"""
import math
import os
import numpy as np
# 系数权重
u = 0.7
rho = 0.3

def read_points_from_file(filename):
    filename = os.path.join('./bank/posturebank', filename)
    points = []
    with open(filename, 'r') as file:
        for line in file:
            x, y = line.strip().split(',')
            points.append((float(x), float(y)))
    return points

def calculate_angle(A, B, C):
    # 计算向量 AB 和 BC
    AB = (B[0] - A[0], B[1] - A[1])
    BC = (C[0] - B[0], C[1] - B[1])

    # 计算向量 AB 和 BC 的内积
    dot_product = AB[0] * BC[0] + AB[1] * BC[1]

    # 计算向量 AB 和 BC 的模长
    length_AB = math.sqrt(AB[0] ** 2 + AB[1] ** 2)
    length_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)

    # 计算角度的弧度值
    angle_radians = math.acos(dot_product / (length_AB * length_BC))

    # 将弧度转换为角度
    angle_degrees = math.degrees(angle_radians)
    
    # 检查角度是否接近于零
    if abs(angle_degrees) < 1e-6:
        angle_degrees = 0.0
        
    return angle_degrees

def calculate_take_off_angles(points):
    # 用归一化的角来计算
    angles = []
    theta1 = calculate_angle(points[9], points[11], points[13])
    angles.append(theta1)
    temp_point = (points[10][0]+ 1, points[10][1])
    theta2 = calculate_angle(points[12], points[10], temp_point)
    angles.append(theta2)
    theta3 = calculate_angle(points[10], points[12], points[14])
    angles.append(theta3)
    return angles

def calculate_hip_extension_angles(points):
    # 用归一化的角来计算
    angles = []

    theta1 = calculate_angle(points[3], points[5], points[7])      
    theta1_ = calculate_angle(points[4], points[6], points[8])
    if(points[1][0] < points[7][0]):
        # 左右手臂夹角
        theta1=theta1*2
        theta1_=theta1_*2
    angles.append(theta1)
    angles.append(theta1_)
    # 躯干与竖直方向的夹角
    temp_point = (points[2][0], points[2][1] - 1)
    theta2 = calculate_angle(points[1], points[2], temp_point)
    angles.append(theta2)
   
#    左右膝关节与髋关节连线与竖直方向的夹角

    temp_point = (points[9][0], points[9][1] + 1)
    theta3 = calculate_angle(points[11], points[9], temp_point)
        
    temp_point = (points[10][0], points[10][1] + 1)
    theta3_ = calculate_angle(points[12], points[10], temp_point)
    if points[11][0] > points[2][0]:
        theta3=theta3*2
    if points[12][0] > points[2][0]:
        theta3_=theta3_*2
    angles.append(theta3)
    angles.append(theta3_)

    return angles

def calculate_abdominal_contraction_angles(points):
    # 用归一化的角来计算
    angles = []
    theta1 = calculate_angle(points[3], points[9], points[11])
    angles.append(theta1)
    theta1_ = calculate_angle(points[4], points[10], points[12])
    angles.append(theta1_)
    
    theta2 = calculate_angle(points[9], points[11], points[13])
    angles.append(theta2)
    theta2_ = calculate_angle(points[10], points[12], points[14])
    angles.append(theta2_)
    
    theta3 = calculate_angle(points[3], points[5], points[7])
    angles.append(theta3)
    theta3_ = calculate_angle(points[4], points[6], points[8])
    
    angles.append(theta3_)
    return angles

def calculate_distance(test_points,posture_type, weights, threshold):
    percentages = None
    # 权重自设
    if posture_type == 'take_off':
        filename = posture_type + '.txt'
        standard_points = read_points_from_file(filename)
        p = calculate_take_off_angles(test_points) # 测试姿态角向量合集
        q = calculate_take_off_angles(standard_points) # 标准姿态角向量合集
        
    if posture_type == 'hip_extension':
        filename = posture_type + '.txt'
        standard_points = read_points_from_file(filename)
        p = calculate_hip_extension_angles(test_points) 
        q = calculate_hip_extension_angles(standard_points) 
    
    if posture_type == 'abdominal_contraction':
        filename = posture_type + '.txt'
        standard_points = read_points_from_file(filename)
        p = calculate_abdominal_contraction_angles(test_points) # 测试姿态角向量合集
        q = calculate_abdominal_contraction_angles(standard_points) # 标准姿态角向量合集
        
    # 计算加权距离
    d = math.sqrt(sum((x - y) ** 2 * w for x, y, w in zip(p, q, weights)))
    
    if d < threshold:
        dot_product = np.dot(p, q)  # 计算点积
        magnitude_p = np.linalg.norm(p)  # 计算向量 p 的模长
        magnitude_q = np.linalg.norm(q)  # 计算向量 q 的模长

        cosine_distance = dot_product / (magnitude_p * magnitude_q)  # 计算余弦距离
        d_cos = 1 - cosine_distance
        
        d = u * d + rho * d_cos
    percentages = calculate_individual_cosine_percentages(p, q)    
    return d,percentages

def calculate_individual_cosine_percentages(p, q):
    cosine_percentages = []
    for angle_p, angle_q in zip(p, q):
        rad_p = math.radians(angle_p)
        rad_q = math.radians(angle_q)
        cos_p = math.cos(rad_p)
        sin_p = math.sin(rad_p)
        cos_q = math.cos(rad_q)
        sin_q = math.sin(rad_q)
        
        dot_product = cos_p * cos_q + sin_p * sin_q
        magnitude_p = math.sqrt(cos_p**2 + sin_p**2)
        magnitude_q = math.sqrt(cos_q**2 + sin_q**2)
        
        cosine_similarity = dot_product / (magnitude_p * magnitude_q)
        # 将余弦相似度转换为百分比
        cosine_percentage = (cosine_similarity + 1) / 2 * 100  # 将[-1, 1]映射到[0, 100]
        cosine_percentages.append(cosine_percentage)
        # print("cosine_percentage:",cosine_percentage)
    
    return cosine_percentages