# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
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
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (QApplication, QDialog, QPushButton, QSizePolicy,
    QWidget)

class Ui_Main(object):
    def setupUi(self, Main):
        if not Main.objectName():
            Main.setObjectName(u"Main")
        Main.resize(1600, 900)
        self.controlButton = QPushButton(Main)
        self.controlButton.setObjectName(u"controlButton")
        self.controlButton.setGeometry(QRect(760, 340, 80, 40))
        self.selectVideoButton = QPushButton(Main)
        self.selectVideoButton.setObjectName(u"selectVideoButton")
        self.selectVideoButton.setGeometry(QRect(760, 410, 80, 40))
        self.editButton = QPushButton(Main)
        self.editButton.setObjectName(u"editButton")
        self.editButton.setGeometry(QRect(1280, 620, 80, 40))
        self.originVideo = QVideoWidget(Main)
        self.originVideo.setObjectName(u"originVideo")
        self.originVideo.setGeometry(QRect(90, 220, 640, 360))
        self.keypointVideo = QVideoWidget(Main)
        self.keypointVideo.setObjectName(u"keypointVideo")
        self.keypointVideo.setGeometry(QRect(860, 220, 640, 360))
        self.buildButton = QPushButton(Main)
        self.buildButton.setObjectName(u"buildButton")
        self.buildButton.setGeometry(QRect(1050, 620, 80, 40))

        self.retranslateUi(Main)

        QMetaObject.connectSlotsByName(Main)
    # setupUi

    def retranslateUi(self, Main):
        Main.setWindowTitle(QCoreApplication.translate("Main", u"Dialog", None))
        self.controlButton.setText(QCoreApplication.translate("Main", u"\u6682\u505c", None))
        self.selectVideoButton.setText(QCoreApplication.translate("Main", u"\u9009\u62e9\u89c6\u9891", None))
        self.editButton.setText(QCoreApplication.translate("Main", u"\u5236\u4f5c\u8bc4\u4ef7\u65b9\u6848", None))
        self.buildButton.setText(QCoreApplication.translate("Main", u"\u751f\u6210\u70b9\u56fe", None))
    # retranslateUi

