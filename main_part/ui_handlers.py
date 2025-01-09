import os
from PySide6.QtUiTools import QUiLoader
from collections import namedtuple
from PySide6.QtWidgets import QPushButton, QLabel, QTableWidget, QInputDialog, QFileDialog


# 界面ui预加载
def setup_ui(main_window):
    """ 加载 UI 文件并初始化控件 """
    ui_file = os.path.join('public', 'resources', 'newwindow', 'Newmain.ui')
    main_window.ui = QUiLoader().load(ui_file, main_window)

    # 定义控件元组
    Control = namedtuple('Control', ['name', 'type'])
    controls = [
        Control('Ca_main', QLabel), Control('Ca_photo', QLabel),
        Control('Preview', QPushButton), Control('Star', QPushButton),
        Control('switch_run', QPushButton),
        Control('switch_jump', QPushButton), Control('results_but', QPushButton),
        Control('Star_test_but', QPushButton), Control('load_stuscoreBut', QPushButton),
        Control('load_speed_video_but', QPushButton), Control('load_jump_video_but', QPushButton),
        Control('load_stumessageBut', QPushButton), 
        Control('set_calibration_but', QPushButton), Control('Ca_run_but', QPushButton),
        Control('Ca_jump_but', QPushButton), Control('Ca_landing_but', QPushButton),
        Control('timerLabel', QLabel), 
        Control('studentList_1', QTableWidget), Control('studentList_2', QTableWidget),
        Control('studentList_3', QTableWidget),
        Control('name_label', QLabel), Control('sno_label', QLabel), 
        Control('Ca_calibration', QLabel),
        Control('but_1', QPushButton),
        Control('but_2', QPushButton),
        Control('but_3', QPushButton),
        Control('load_standardBut', QPushButton),
        Control('labelMeanVector', QLabel),
        Control('labelTakeOffScore', QLabel),
        Control('labelHipExtensionScore', QLabel),
        Control('labelAbdominalContractionScore', QLabel),
        Control('labelAllScore', QLabel),
    ]

    # 动态查找并设置控��为类属性
    for control in controls:
        widget = main_window.ui.findChild(control.type, control.name)
        if widget is None:
            print(f"无法找到控件: {control.name}")
        setattr(main_window, control.name, widget)

    # 初始化得分显示为 0
    try:
        main_window.labelMeanVector.setText("0.00 m/s")
        main_window.labelTakeOffScore.setText("0.00")
        main_window.labelHipExtensionScore.setText("0.00")
        main_window.labelAbdominalContractionScore.setText("0.00")
        main_window.labelAllScore.setText("0.00")
    except Exception as e:
        print(f"初始化标签显示时出错: {str(e)}")
        import traceback
        traceback.print_exc()


def connect_signals(main_window):
    """ 自动连接按钮与槽函数 """
    signal_map = {
        'Preview': main_window.toggle_cameras,
        'Star': main_window.star_Recording,
        'results_but': main_window.results,
        'Star_test_but': main_window.Star_test,
        'load_stuscoreBut': main_window.load_stuscore,
        'load_stumessageBut': main_window.loadstudents,
        'switch_run': main_window.show_camera1_view,
        'switch_jump': main_window.show_camera2_view,
        'Ca_run_but': main_window.show_runvideo_preview,
        'Ca_jump_but': main_window.show_jumpvideo_preview,
        'set_calibration_but': main_window.show_calibration_dialog,
        'load_speed_video_but': main_window.load_speed_video,
        'load_jump_video_but': main_window.load_jump_video,
        'load_standardBut':main_window.load_standardBut_xx,
        'but_1': lambda: main_window.switch_test_number(1),
        'but_2': lambda: main_window.switch_test_number(2),
        'but_3': lambda: main_window.switch_test_number(3),
       
    }

    # 自动连接按钮与对应处理函数
    for button_name, handler in signal_map.items():
        button = getattr(main_window, button_name, None)
        if isinstance(button, QPushButton) and callable(handler):
            button.clicked.connect(handler)
        else:
            print(f"连接信号失败: 控件 '{button_name}' 或处理函数 '{handler}' 无效")

    # 连接 studentList 相关的点击事件
    main_window.studentList_1.itemClicked.connect(main_window.on_student_clicked)
    main_window.studentList_2.itemClicked.connect(main_window.on_student_clicked)
    main_window.studentList_3.itemClicked.connect(main_window.on_student_clicked)
    
# 额外的 UI 操作逻辑函数

def select_video_source(main_window):
    """ 弹出文件选择器，选择视频文件 """
    video_path, _ = QFileDialog.getOpenFileName(main_window, "选择视频文件", "", "视频文件 (*.avi *.mp4)")
    return video_path

def set_board_distance(main_window):
    """ 弹出输入框，设置参考物与起跳板的距离 """
    distance, ok = QInputDialog.getInt(main_window, "设置参考物与起跳版距离", "请输入距离：")
    if ok:
        main_window.runway.reference_to_jump_distance = distance
        main_window.runway.changeconfig("reference_to_jump_distance", distance)
        print(f"设置的距离为: {distance}")
    return ok
