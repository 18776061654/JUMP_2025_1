# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'evaluation.ui'
##
## Created by: Qt User Interface Compiler version 6.6.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QLabel,
    QPushButton, QScrollArea, QSizePolicy, QSlider,
    QWidget)

class Ui_Evaluation(object):
    def setupUi(self, Evaluation):
        if not Evaluation.objectName():
            Evaluation.setObjectName(u"Evaluation")
        Evaluation.resize(1400, 820)
        self.preFrame = QPushButton(Evaluation)
        self.preFrame.setObjectName(u"preFrame")
        self.preFrame.setGeometry(QRect(350, 730, 90, 40))
        self.nextFrame = QPushButton(Evaluation)
        self.nextFrame.setObjectName(u"nextFrame")
        self.nextFrame.setGeometry(QRect(520, 730, 91, 41))
        self.editButton = QPushButton(Evaluation)
        self.editButton.setObjectName(u"editButton")
        self.editButton.setGeometry(QRect(790, 730, 90, 40))
        self.save_button = QPushButton(Evaluation)
        self.save_button.setObjectName(u"save_button")
        self.save_button.setGeometry(QRect(910, 730, 90, 40))
        self.keypointPic = QLabel(Evaluation)
        self.keypointPic.setObjectName(u"keypointPic")
        self.keypointPic.setGeometry(QRect(130, 50, 1120, 630))
        self.keypointPic.setFrameShape(QFrame.Box)
        self.keypointPic.setAlignment(Qt.AlignCenter)
        self.scaleSlider = QSlider(Evaluation)
        self.scaleSlider.setObjectName(u"scaleSlider")
        self.scaleSlider.setGeometry(QRect(1310, 60, 22, 630))
        self.scaleSlider.setOrientation(Qt.Vertical)
        self.scrollArea = QScrollArea(Evaluation)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setGeometry(QRect(130, 50, 1120, 630))
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 1118, 628))
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.retranslateUi(Evaluation)

        QMetaObject.connectSlotsByName(Evaluation)
    # setupUi

    def retranslateUi(self, Evaluation):
        Evaluation.setWindowTitle(QCoreApplication.translate("Evaluation", u"Dialog", None))
        self.preFrame.setText(QCoreApplication.translate("Evaluation", u"\u4e0a\u4e00\u5e27", None))
        self.nextFrame.setText(QCoreApplication.translate("Evaluation", u"\u4e0b\u4e00\u5e27", None))
        self.editButton.setText(QCoreApplication.translate("Evaluation", u"\u4fee\u6539\u70b9\u4f4d\u7f6e", None))
        self.save_button.setText(QCoreApplication.translate("Evaluation", u"\u4fdd\u5b58\u8be5\u59ff\u52bf", None))
        self.keypointPic.setText("")
    # retranslateUi

