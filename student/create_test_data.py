import pandas as pd
import numpy as np
import os

# 创建测试数据
def create_test_data():
    # 创建20个学生的数据
    data = {
        'student_id': [f'2022010500{str(i).zfill(2)}' for i in range(1, 21)],
        'name': [f'学生{i}' for i in range(1, 21)],
        'gender': np.random.choice(['男', '女'], size=20),  # 保留但不显示
        'class': np.random.choice(['计算机1班', '计算机2班', '计算机3班'], size=20),  # 保留但不显示
        'test_count': [0] * 20,  # 初始测评次数都为0
        'best_score': [0.0] * 20,  # 保留但不显示
        'status': ['未测评'] * 20  # 初始状态都是未测评
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 确保student目录存在
    os.makedirs('student', exist_ok=True)
    
    # 保存到Excel文件
    output_path = os.path.join('student', 'test_students.xlsx')
    df.to_excel(output_path, index=False)
    print(f"测试数据已生成到: {output_path}")

if __name__ == "__main__":
    create_test_data() 