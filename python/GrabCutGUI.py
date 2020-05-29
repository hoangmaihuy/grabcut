import sys
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QGraphicsScene
from PyQt5.QtGui import QPixmap, QPen, QColor, QPainterPath, QBrush
from PyQt5.QtCore import QRectF, QLineF, QPointF
from GrabCut import GrabCut, Trimap
from GrabCutQtDesignerUI import Ui_MainWindow


class EditMode(IntEnum):
    DEFAULT = 0
    SET_B_REGION = 1
    ADD_B_SEED = 2
    ADD_F_SEED = 3


class Color:
    RED = QColor(255, 0, 0)
    BLUE = QColor(0, 0, 255)


class ImageViewer(QGraphicsScene):
    def __init__(self):
        super(ImageViewer, self).__init__()
        self.BRUSH_RADIUS = 3
        self.imagePath = None
        self.image = None
        self.mode = EditMode.DEFAULT
        self.drawing = False
        self.fromPos = None
        self.toPos = None
        self.mask = None
        self.rect = None
        self.pen = QPen()

    def setMode(self, mode):
        self.mode = mode

    def setImage(self, imagePath):
        self.imagePath = imagePath
        self.addPixmap(QPixmap(imagePath))
        self.image = plt.imread(imagePath)

    def setMask(self, fromPos, toPos, value):
        self.mask[int(fromPos.y()):int(toPos.y()), int(fromPos.x()):int(toPos.x())] = int(value)

    def getPen(self):
        if self.mode == EditMode.DEFAULT:
            return self.pen
        if self.mode == EditMode.ADD_F_SEED:
            self.pen.setColor(Color.BLUE)
        else:
            self.pen.setColor(Color.RED)
        return self.pen

    def mousePressEvent(self, event):
        if self.mode == EditMode.DEFAULT:
            return
        pos = event.scenePos()
        self.drawing = True
        if self.mode == EditMode.SET_B_REGION:
            self.fromPos = pos
            self.rect = self.addRect(QRectF(pos, pos), self.getPen())
            return
        if self.mask is None:
            self.mask = np.full(self.image.shape[:2], Trimap.UKN, np.uint8)
        if self.mode == EditMode.ADD_F_SEED:
            self.setMask(pos, pos, Trimap.FGD)
        elif self.mode == EditMode.ADD_B_SEED:
            self.setMask(pos, pos, Trimap.BGD)
        self.addPath(QPainterPath(pos), self.getPen())

    def mouseMoveEvent(self, event):
        if self.mode == EditMode.DEFAULT or not self.drawing:
            return
        pos = event.scenePos()
        if self.mode == EditMode.SET_B_REGION:
            self.removeItem(self.rect)
            self.rect = self.addRect(QRectF(self.fromPos, pos), self.getPen())
            return
        fromPos = pos - QPointF(self.BRUSH_RADIUS, self.BRUSH_RADIUS)
        toPos = pos + QPointF(self.BRUSH_RADIUS, self.BRUSH_RADIUS)

        value = Trimap.FGD if self.mode == EditMode.ADD_F_SEED else Trimap.BGD
        color = Color.BLUE if self.mode == EditMode.ADD_F_SEED else Color.RED
        self.setMask(fromPos, toPos, value)
        self.addEllipse(QRectF(fromPos, toPos), self.getPen(), QBrush(color))

    def mouseReleaseEvent(self, event):
        if self.mode == EditMode.DEFAULT or not self.drawing:
            return
        pos = event.scenePos()
        self.drawing = False
        if self.mode == EditMode.SET_B_REGION:
            self.toPos = pos
            rect = QRectF(self.fromPos, self.toPos).getCoords()
            self.rect = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
            # self.setMask(self.fromPos, self.toPos, Trimap.B)
            return

class GrabCutGUI(object):
    def __init__(self):
        self.imagePath = None
        self.grabcut = None
        self.app = QApplication(sys.argv)
        self.MainWindow = QMainWindow()
        self.ImageViewer = ImageViewer()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)
        self.ui.graphicsView.setScene(self.ImageViewer)
        self.bindingEvent()
        self.MainWindow.show()
        sys.exit(self.app.exec_())

    def bindingEvent(self):
        self.ui.actionOpen_Image.triggered.connect(self.openImage)
        self.ui.actionSave_Image.triggered.connect(self.saveImage)
        self.ui.actionSet_Background_Region.triggered.connect(self.setBackgroundRegion)
        self.ui.actionAdd_Background_Seed.triggered.connect(self.addBackgroundSeed)
        self.ui.actionAdd_Foreground_Seed.triggered.connect(self.addForegroundSeed)
        self.ui.actionClear_Input.triggered.connect(self.clearInput)
        self.ui.setBackgroundRegionButton.clicked.connect(self.setBackgroundRegion)
        self.ui.addBackgroundSeedButton.clicked.connect(self.addBackgroundSeed)
        self.ui.addForegroundSeedButton.clicked.connect(self.addForegroundSeed)
        self.ui.clearInputButton.clicked.connect(self.clearInput)
        self.ui.runButton.clicked.connect(self.runGrabCut)

    def openImage(self):
        self.imagePath, _ = QFileDialog.getOpenFileName(self.MainWindow, "Open Image", "" ,"Image files (*.jpg)")
        self.ImageViewer.setImage(self.imagePath)
        self.ui.statusbar.showMessage("Opened " + self.imagePath)

    def saveImage(self):
        savePath, _ = QFileDialog.getSaveFileName(self.MainWindow, "Save Image As...", "")
        self.ui.statusbar.showMessage("Saved file to " + savePath)

    def setBackgroundRegion(self):
        self.ui.statusbar.showMessage("Setting Background Region")
        self.ImageViewer.setMode(EditMode.SET_B_REGION)

    def addBackgroundSeed(self):
        self.ui.statusbar.showMessage("Adding Background Seed")
        self.ImageViewer.setMode(EditMode.ADD_B_SEED)

    def addForegroundSeed(self):
        self.ui.statusbar.showMessage("Adding Foreground Region")
        self.ImageViewer.setMode(EditMode.ADD_F_SEED)

    def clearInput(self):
        self.ImageViewer.rect = None
        self.ImageViewer.mask = None
        self.ImageViewer.clear()
        self.grabcut = None

    def runGrabCut(self):
        if self.grabcut is None:
            self.grabcut = GrabCut(self.imagePath)
        resultPath = self.grabcut.run(self.ImageViewer.rect, self.ImageViewer.mask)
        self.imagePath = resultPath
        self.clearInput()
        self.ImageViewer.setImage(resultPath)

if __name__ == '__main__':
    gui = GrabCutGUI()