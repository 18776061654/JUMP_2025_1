import os
import json
import traceback
from PySide6.QtWidgets import QMainWindow, QMessageBox, QTableWidget, QTableWidgetItem
from PySide6.QtGui import QPixmap, QColor
from PySide6.QtCore import QFile, Qt
from PySide6.QtUiTools import QUiLoader

class PoseMatchDetails(QMainWindow):
    def __init__(self, pose_dir, pose_type, student_name=None, student_id=None):
        """
        初始化姿态匹配详细界面
        Args:
            pose_dir: 姿态文件夹路径（如 .../test_1/jump/takeoff）
            pose_type: 姿态类型（'takeoff', 'flight', 'landing'）
            student_name: 学生姓名
            student_id: 学生学号
        """
        super().__init__()
        
        try:
            # 加载UI文件
            ui_file = QFile('public/resources/newwindow/pose_match_details.ui')
            if not ui_file.open(QFile.ReadOnly):
                raise IOError(f"无法打开UI文件")
            
            self.ui = QUiLoader().load(ui_file)
            ui_file.close()
            
            if not self.ui:
                raise RuntimeError("UI加载失败")
            
            # 设置主
            self.setCentralWidget(self.ui)
            self.setWindowTitle(f"姿态匹配详情 - {pose_type}")
            self.resize(1349, 758)
            
            # 存储路径信息
            self.pose_dir = pose_dir
            self.pose_type = pose_type
            
            # 存储评分标准数据
            self.current_page = 0
            self.angles_evaluation = []
            self.items_per_page = 1
            
            # 设置学生信息
            self.ui.name_label.setText(student_name or "N/A")
            self.ui.sno_label.setText(student_id or "N/A")
            
            # 加载得分信息
            self.load_score_info()
            
            # 加载匹配数据和图片
            self.load_match_details()
            
            # 连接翻页按钮信号
            self.ui.prevPageBtn.clicked.connect(self.show_prev_page)
            self.ui.nextPageBtn.clicked.connect(self.show_next_page)
            
            # 存储学生文件夹路径（用于切换姿态）
            self.student_folder = os.path.dirname(os.path.dirname(pose_dir))
            
            # 存储当前姿态信息
            self.pose_types = ['takeoff', 'flight', 'landing']
            self.current_pose_index = self.pose_types.index(pose_type)
            
            # 连接姿态切换按钮
            self.ui.prevPoseBtn.clicked.connect(self.show_prev_pose)
            self.ui.nextPoseBtn.clicked.connect(self.show_next_pose)
            
            # 更新按钮状态
            self.update_pose_buttons()
            
            # 初始显示第一页
            self.load_scoring_criteria()
            
        except Exception as e:
            print(f"初始化姿态匹配详情窗口时出错: {str(e)}")
            traceback.print_exc()

    def load_match_details(self):
        """加载匹配详情数和图片"""
        try:
            # 显示匹配姿态图片
            best_frame_img = os.path.join(self.pose_dir, 'compare_frame_best_cropped.jpg')
            if os.path.exists(best_frame_img):
                self.ui.label_match_posture.setPixmap(QPixmap(best_frame_img).scaled(
                    self.ui.label_match_posture.width(),
                    self.ui.label_match_posture.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
            
            # 显示参考姿态图片
            reference_paths = {
                'takeoff': 'public/data/standard_poses/takeoff/standard.png',
                'flight': 'public/data/standard_poses/flight/standard.png',
                'landing': 'public/data/standard_poses/landing/standard.png'
            }
            
            reference_img = reference_paths.get(self.pose_type)
            if reference_img and os.path.exists(reference_img):
                self.ui.label_reference_posture.setPixmap(QPixmap(reference_img).scaled(
                    self.ui.label_reference_posture.width(),
                    self.ui.label_reference_posture.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
            
            # 设置关键点角度表格
            self.setup_angle_table()
            
        except Exception as e:
            print(f"加载匹配详情时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def setup_angle_table(self):
        """设置关键点角度表格"""
        try:
            # 设置表格大小
            self.ui.tableAngles.setFixedSize(433, 273)
            
            # 禁用滚动条
            self.ui.tableAngles.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.ui.tableAngles.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            
            # 设置表头
            self.ui.tableAngles.setColumnCount(4)
            headers = ["关键角度", "结果(°)", "推荐值(°)", "得分"]
            self.ui.tableAngles.setHorizontalHeaderLabels(headers)
            
            # 隐藏左侧序号列
            self.ui.tableAngles.verticalHeader().setVisible(False)
            
            # 获取当前阶段的评分数据
            score_file = os.path.join(os.path.dirname(self.pose_dir), 'jump_score_result.json')
            if os.path.exists(score_file):
                with open(score_file, 'r', encoding='utf-8') as f:
                    score_data = json.load(f)
                    
                # 找到当前阶段的评分数据
                phase_eval = next((phase for phase in score_data['phase_evaluations'] 
                                 if phase['phase'] == self.pose_type), None)
                
                if phase_eval and 'angles_evaluation' in phase_eval:
                    angles_data = phase_eval['angles_evaluation']
                    
                    # 设置行数
                    self.ui.tableAngles.setRowCount(len(angles_data))
                    
                    # 计算每行的高度
                    available_height = 273 - self.ui.tableAngles.horizontalHeader().height()
                    row_height = available_height // len(angles_data)
                    
                    # 填充数据
                    for row, angle_data in enumerate(angles_data):
                        # 设置行高
                        self.ui.tableAngles.setRowHeight(row, row_height)
                        
                        # 关键角度名称
                        self.ui.tableAngles.setItem(row, 0, 
                            QTableWidgetItem(angle_data['angle_description']))
                        
                        # 实际角度
                        self.ui.tableAngles.setItem(row, 1, 
                            QTableWidgetItem(f"{angle_data['actual_angle']:.1f}"))
                        
                        # 推荐角度
                        self.ui.tableAngles.setItem(row, 2, 
                            QTableWidgetItem(f"{angle_data['recommended_angle']}"))
                        
                        # 得分
                        self.ui.tableAngles.setItem(row, 3, 
                            QTableWidgetItem(f"{angle_data['score']:.1f}"))
                    
                    # 设置列宽
                    total_width = self.ui.tableAngles.width()
                    column_widths = [total_width * 0.4, total_width * 0.2, total_width * 0.2, total_width * 0.2]  # 40%, 20%, 20%, 20%
                    for col, width in enumerate(column_widths):
                        self.ui.tableAngles.setColumnWidth(col, int(width))
                    
                    # 设置表格样式
                    self.ui.tableAngles.setStyleSheet("""
                        QTableWidget {
                            background-color: white;
                            border: none;
                            gridline-color: transparent;  /* 取消网格线 */
                        }
                        QTableWidget::item {
                            padding: 5px;
                            border: none;  /* 取消单元格边框 */
                            font-size: 20px;  /* 增大单元格内容字体 */
                        }
                        QHeaderView::section {
                            background-color: #2c3e50;
                            color: white;
                            padding: 5px;
                            border: none;
                            font-size: 18px;  /* 增大表头字体 */
                            font-weight: bold;
                        }
                        QTableWidget::item:selected {
                            background-color: transparent;
                        }
                        /* 取消表头和单元格之间的边框 */
                        QHeaderView {
                            border: none;
                        }
                        QTableCornerButton::section {
                            border: none;
                            background-color: #2c3e50;
                        }
                        /* 隐藏滚动条 */
                        QScrollBar {
                            width: 0px;
                            height: 0px;
                        }
                    """)
                    
                    # 设置表格内容居中
                    for row in range(self.ui.tableAngles.rowCount()):
                        for col in range(self.ui.tableAngles.columnCount()):
                            item = self.ui.tableAngles.item(row, col)
                            if item:
                                item.setTextAlignment(Qt.AlignCenter)
                    
                    # 禁用选择
                    self.ui.tableAngles.setSelectionMode(QTableWidget.NoSelection)
                    self.ui.tableAngles.setFocusPolicy(Qt.NoFocus)
                    
        except Exception as e:
            print(f"设置角度表格时出错: {str(e)}")
            traceback.print_exc()

    def load_scoring_criteria(self):
        """加载评分标准"""
        try:
            score_file = os.path.join(self.pose_dir, f'{self.pose_type}_score_result.json')
            if os.path.exists(score_file):
                with open(score_file, 'r', encoding='utf-8') as f:
                    score_data = json.load(f)
                    self.angles_evaluation = score_data.get('angles_evaluation', [])
                    
                    # 显示当前页的评分标准
                    self.show_current_page()
                    
                    # 显示建议内容
                    self.show_suggestions(score_data)
                    
                    # 更新页码显示
                    total_pages = len(self.angles_evaluation)
                    self.ui.pageLabel.setText(f"第 {self.current_page + 1}/{total_pages} 页")
                    
                    # 更新按钮状态
                    self.update_button_states()
            
        except Exception as e:
            print(f"加载评分标准时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def show_current_page(self):
        """显示当前页的评分标准"""
        try:
            if self.current_page < len(self.angles_evaluation):
                angle_data = self.angles_evaluation[self.current_page]
                
                html_content = """
                <style>
                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }
                    body {
                        width: 325px;  /* 设置为固定宽度 */
                        font-family: Microsoft YaHei;
                    }
                    .container {
                        width: 100%;
                        padding: 2px;
                    }
                    .title {
                        font-size: 15px;
                        font-weight: bold;
                        color: #333;
                        text-align: center;
                        margin: 2px 0;
                        padding: 3px;
                        background-color: #f0f0f0;
                    }
                    table { 
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 2px;
                    }
                    th { 
                        background-color: #2c3e50;
                        color: white;
                        padding: 4px 2px;
                        font-size: 13px;
                        text-align: center;
                    }
                    td { 
                        padding: 4px 2px;
                        font-size: 13px;
                        text-align: center;
                    }
                    .excellent { background-color: #90EE90; }
                    .good { background-color: #FFFFE0; }
                    .poor { background-color: #FFC0C0; }
                </style>
                <div class="container">
                """
                
                # 添加度名称
                angle_name = angle_data.get('angle_description', '')
                html_content += f'<div class="title">{angle_name}</div>'
                
                # 添加表格
                html_content += """
                <table cellspacing="0" cellpadding="0">
                    <tr>
                        <th width="34%">角度范围</th>
                        <th width="33%">得分范围</th>
                        <th width="33%">评价</th>
                    </tr>
                """
                
                # 添加评分范围
                for range_info in angle_data.get('score_ranges', []):
                    start, end = range_info['range']
                    min_score = range_info.get('min_score', 0)
                    max_score = range_info['score']
                    
                    if max_score >= 80:
                        style_class = 'excellent'
                        evaluation = '优秀'
                    elif max_score >= 60:
                        style_class = 'good'
                        evaluation = '良好'
                    else:
                        style_class = 'poor'
                        evaluation = '需要改进'
                    
                    html_content += f"""
                    <tr class='{style_class}'>
                        <td>{start}° - {end}°</td>
                        <td>{min_score}-{max_score}分</td>
                        <td>{evaluation}</td>
                    </tr>
                    """
                
                html_content += "</table></div>"
                
                # 设置QTextBrowser的样式
                self.ui.textScoring.setStyleSheet("""
                    QTextBrowser {
                        background-color: white;
                        border: none;
                        padding: 0px;
                        margin: 0px;
                    }
                    QTextBrowser QScrollBar {
                        width: 0px;  /* 隐藏滚动条 */
                    }
                """)
                
                # 显示内容
                self.ui.textScoring.setHtml(html_content)
                
                # 更新页码显示
                total_pages = len(self.angles_evaluation)
                self.ui.pageLabel.setText(f"第 {self.current_page + 1}/{total_pages} 页")
                
        except Exception as e:
            print(f"显示评分标准时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def show_prev_page(self):
        """显示上一页"""
        if self.current_page > 0:
            self.current_page -= 1
            self.show_current_page()
            self.update_button_states()
            # 更新页码显示
            total_pages = len(self.angles_evaluation)  # 直接使用角度数量
            self.ui.pageLabel.setText(f"第 {self.current_page + 1}/{total_pages} 页")

    def show_next_page(self):
        """显示下一页"""
        total_pages = len(self.angles_evaluation)  # 直接使用角度数量
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.show_current_page()
            self.update_button_states()
            # 更新页码显示
            self.ui.pageLabel.setText(f"第 {self.current_page + 1}/{total_pages} 页")

    def update_button_states(self):
        """更新按钮状态"""
        total_pages = len(self.angles_evaluation)  # 直接使用角度数量
        self.ui.prevPageBtn.setEnabled(self.current_page > 0)
        self.ui.nextPageBtn.setEnabled(self.current_page < total_pages - 1) 

    def show_suggestions(self, score_data):
        """显示评分建议"""
        try:
            # 获取建议内容
            suggestions = []
            
            # 收集所有需要改进的角度的建议
            for angle_data in self.angles_evaluation:
                score = angle_data.get('score', 0)
                angle_name = angle_data.get('angle_description', '')
                actual_angle = angle_data.get('actual_angle', 0)
                recommended_angle = angle_data.get('recommended_angle', 0)
                
                if score < 60:
                    if actual_angle > recommended_angle:
                        suggestions.append(f"• {angle_name}角度过大，建议减小至{recommended_angle}°左右")
                    else:
                        suggestions.append(f"• {angle_name}角度过小，建议增大至{recommended_angle}°左右")
                elif score < 80:
                    suggestions.append(f"• {angle_name}角度尚可，但还可以整至{recommended_angle}°左右以获得更好的效果")
            
            # 如果没有具体建议，添加一个默认的积极评价
            if not suggestions:
                suggestions.append("• 各个关键点角度控制得当，继续保持！")
            
            # 设置建议文本的样式
            html_content = """
            <style>
                body {
                    font-family: Microsoft YaHei;
                    font-size: 14px;
                    line-height: 1.5;
                    color: #333;
                    margin: 10px;
                }
                .suggestion-title {
                    font-weight: bold;
                    margin-bottom: 8px;
                    color: #2c3e50;
                }
                .suggestion-list {
                    margin-left: 5px;
                }
            </style>
            <div class="suggestion-title">动作建议：</div>
            <div class="suggestion-list">
            """
            
            # 添加所有建议
            html_content += "\n".join(suggestions)
            html_content += "</div>"
            
            # 设置文本显示
            self.ui.textSuggestions.setHtml(html_content)
            
        except Exception as e:
            print(f"显示建议时出错: {str(e)}")
            import traceback
            traceback.print_exc() 

    def load_score_info(self):
        """加载得分信息"""
        try:
            # 获取得分文件路径
            score_file = os.path.join(os.path.dirname(self.pose_dir), 'jump_score_result.json')
            
            if os.path.exists(score_file):
                with open(score_file, 'r', encoding='utf-8') as f:
                    score_data = json.load(f)
            
            # 构建显示文本
            html_content = """
            <style>
                body {
                    background: transparent;
                }
                .phase-score {
                    font-family: Arial;
                    font-size: 48px;
                    font-weight: bold;
                    color: white;
                    margin-bottom: 10px;
                    text-align: center;
                }
                .evaluation {
                    font-family: Microsoft YaHei;
                    font-size: 14px;
                    color: white;
                    text-align: center;
                }
            </style>
            """
            
            # 添加阶段得分和评价
            phase_score = score_data['phase_scores'][self.pose_type]['score']
            phase_eval = next((phase for phase in score_data['phase_evaluations'] 
                             if phase['phase'] == self.pose_type), None)
            
            if phase_eval:
                html_content += f"""
                <div class='phase-score'>{phase_score:.2f}</div>
                <div class='evaluation'>{phase_eval['evaluation']}</div>
                """
            
            # 设置文本显示
            if hasattr(self.ui, 'labelkeyScore'):
                self.ui.labelkeyScore.setHtml(html_content)
            else:
                print("警告: 未找到labelkeyScore控件")
            
        except Exception as e:
            print(f"加载得分信息时出错: {str(e)}")
            traceback.print_exc()

    def get_phase_name(self, phase_type):
        """获取阶段的中文名称"""
        phase_names = {
            'takeoff': '起跳阶段',
            'flight': '空中阶段',
            'landing': '落地阶段'
        }
        return phase_names.get(phase_type, phase_type) 

    def show_prev_pose(self):
        """显示上一个姿态"""
        if self.current_pose_index > 0:
            self.current_pose_index -= 1
            new_pose_type = self.pose_types[self.current_pose_index]
            new_pose_dir = os.path.join(self.student_folder, 'jump', new_pose_type)
            
            # 更新窗口标题
            self.setWindowTitle(f"姿态匹配详情 - {self.get_phase_name(new_pose_type)}")
            
            # 更新当前姿态类型和路径
            self.pose_type = new_pose_type
            self.pose_dir = new_pose_dir
            
            # 重新加载所有数据
            self.load_score_info()
            self.load_match_details()
            self.load_scoring_criteria()
            
            # 更新按钮状态
            self.update_pose_buttons()

    def show_next_pose(self):
        """显示下一个姿态"""
        if self.current_pose_index < len(self.pose_types) - 1:
            self.current_pose_index += 1
            new_pose_type = self.pose_types[self.current_pose_index]
            new_pose_dir = os.path.join(self.student_folder, 'jump', new_pose_type)
            
            # 更新窗口标题
            self.setWindowTitle(f"姿态匹配详情 - {self.get_phase_name(new_pose_type)}")
            
            # 更新当前姿态类型和路径
            self.pose_type = new_pose_type
            self.pose_dir = new_pose_dir
            
            # 重新加载所有数据
            self.load_score_info()
            self.load_match_details()
            self.load_scoring_criteria()
            
            # 更新按钮状态
            self.update_pose_buttons()

    def update_pose_buttons(self):
        """更新姿态切换按钮的状态"""
        # 禁用/启用上一个按钮
        self.ui.prevPoseBtn.setEnabled(self.current_pose_index > 0)
        # 禁用/启用下一个按钮
        self.ui.nextPoseBtn.setEnabled(self.current_pose_index < len(self.pose_types) - 1) 