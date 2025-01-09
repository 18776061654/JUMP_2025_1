import os
from PySide6.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QPushButton
from PySide6.QtCore import Slot
import openpyxl
from main_part.show_results import ShowResults
class StudentInfoTable(QMainWindow):
    def __init__(self,studentfile_path):
        super().__init__()
        self.setWindowTitle('学生信息表')
        self.setGeometry(100, 100, 800, 400)  # 设置窗口位置和大小
        self.setup_table(studentfile_path)
        self.results_folder = './results'
    def setup_table(self, studentfile_path):
        wb = openpyxl.load_workbook(studentfile_path)
        ws = wb.active
        data = []
        for row in ws.iter_rows(values_only=True):
            data.append(list(row))

        table = QTableWidget()
        table.setColumnCount(len(data[0]) + 1)  # 添加一个额外的列
        headers = data[0] + ["操作"]
        table.setHorizontalHeaderLabels(headers)

        table.setRowCount(len(data) - 1)
        for i, row in enumerate(data[1:]):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                table.setItem(i, j, item)

            status_index = data[0].index("处理状态")
            if row[status_index] == "已处理":
                button = QPushButton("查看详情")
                # 使用 lambda 将行索引作为参数传递
                button.clicked.connect(lambda checked=False, row=i: self.view_details(row))

                table.setCellWidget(i, len(data[0]), button)

        layout = QVBoxLayout()
        layout.addWidget(table)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)



    @Slot()
    def view_details(self, row_index):
        # 假设学号位于第一列
        table = self.centralWidget().layout().itemAt(0).widget()
        student_id_item = table.item(row_index, 1)  # 获取第二列的单元格项目
        student_id = student_id_item.text()
        print(f"学号: {student_id}")  # 显示学号
        # 在此处添加更多逻辑以处理学号相关的操作
        person_file_name = f"{student_id}"
        self.sele_person_file=os.path.join(self.results_folder,person_file_name)
        self.show_result = ShowResults(self.sele_person_file)
        self.show_result.show()

def main():
    app = QApplication([])
    window = StudentInfoTable()
    window.show()
    app.exec()

if __name__ == '__main__':
    main()