# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'grabcut.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1394, 856)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(130, 10, 1251, 781))
        self.graphicsView.setMouseTracking(True)
        self.graphicsView.setObjectName("graphicsView")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 111, 211))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.setBackgroundRegionButton = QtWidgets.QPushButton(self.groupBox)
        self.setBackgroundRegionButton.setGeometry(QtCore.QRect(10, 10, 93, 28))
        self.setBackgroundRegionButton.setObjectName("setBackgroundRegionButton")
        self.addBackgroundSeedButton = QtWidgets.QPushButton(self.groupBox)
        self.addBackgroundSeedButton.setGeometry(QtCore.QRect(10, 50, 93, 28))
        self.addBackgroundSeedButton.setObjectName("addBackgroundSeedButton")
        self.addForegroundSeedButton = QtWidgets.QPushButton(self.groupBox)
        self.addForegroundSeedButton.setGeometry(QtCore.QRect(10, 90, 93, 28))
        self.addForegroundSeedButton.setObjectName("addForegroundSeedButton")
        self.clearInputButton = QtWidgets.QPushButton(self.groupBox)
        self.clearInputButton.setGeometry(QtCore.QRect(10, 170, 93, 28))
        self.clearInputButton.setObjectName("clearInputButton")
        self.runButton = QtWidgets.QPushButton(self.groupBox)
        self.runButton.setGeometry(QtCore.QRect(10, 130, 93, 28))
        self.runButton.setObjectName("runButton")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 240, 111, 141))
        self.groupBox_2.setObjectName("groupBox_2")
        self.checkBox = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox.setGeometry(QtCore.QRect(10, 20, 81, 20))
        self.checkBox.setObjectName("checkBox")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(20, 50, 55, 16))
        self.label.setText("")
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1394, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setStatusTip("")
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_Image = QtWidgets.QAction(MainWindow)
        self.actionOpen_Image.setObjectName("actionOpen_Image")
        self.actionSave_Image = QtWidgets.QAction(MainWindow)
        self.actionSave_Image.setObjectName("actionSave_Image")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.actionSet_Background_Region = QtWidgets.QAction(MainWindow)
        self.actionSet_Background_Region.setObjectName("actionSet_Background_Region")
        self.actionAdd_Background_Seed = QtWidgets.QAction(MainWindow)
        self.actionAdd_Background_Seed.setObjectName("actionAdd_Background_Seed")
        self.actionAdd_Foreground_Seed = QtWidgets.QAction(MainWindow)
        self.actionAdd_Foreground_Seed.setObjectName("actionAdd_Foreground_Seed")
        self.actionClear_Input = QtWidgets.QAction(MainWindow)
        self.actionClear_Input.setObjectName("actionClear_Input")
        self.actionGMMs_Components = QtWidgets.QAction(MainWindow)
        self.actionGMMs_Components.setObjectName("actionGMMs_Components")
        self.menuFile.addAction(self.actionOpen_Image)
        self.menuFile.addAction(self.actionSave_Image)
        self.menuEdit.addAction(self.actionSet_Background_Region)
        self.menuEdit.addAction(self.actionAdd_Background_Seed)
        self.menuEdit.addAction(self.actionAdd_Foreground_Seed)
        self.menuEdit.addAction(self.actionClear_Input)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "GrabCut"))
        self.setBackgroundRegionButton.setText(_translate("MainWindow", "Set B Region"))
        self.addBackgroundSeedButton.setText(_translate("MainWindow", "Add B Seed"))
        self.addForegroundSeedButton.setText(_translate("MainWindow", "Add F Seed"))
        self.clearInputButton.setText(_translate("MainWindow", "Clear"))
        self.runButton.setText(_translate("MainWindow", "Run"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Config"))
        self.checkBox.setText(_translate("MainWindow", "OpenCV"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.actionOpen_Image.setText(_translate("MainWindow", "Open.."))
        self.actionOpen_Image.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionSave_Image.setText(_translate("MainWindow", "Save as.."))
        self.actionSave_Image.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
        self.actionQuit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.actionSet_Background_Region.setText(_translate("MainWindow", "Set Background Region"))
        self.actionSet_Background_Region.setShortcut(_translate("MainWindow", "Ctrl+R"))
        self.actionAdd_Background_Seed.setText(_translate("MainWindow", "Add Background Seed"))
        self.actionAdd_Background_Seed.setShortcut(_translate("MainWindow", "Ctrl+B"))
        self.actionAdd_Foreground_Seed.setText(_translate("MainWindow", "Add Foreground Seed"))
        self.actionAdd_Foreground_Seed.setShortcut(_translate("MainWindow", "Ctrl+F"))
        self.actionClear_Input.setText(_translate("MainWindow", "Clear Input"))
        self.actionClear_Input.setShortcut(_translate("MainWindow", "Ctrl+K"))
        self.actionGMMs_Components.setText(_translate("MainWindow", "GMMs Components"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
