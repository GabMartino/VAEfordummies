# importing Qt widgets
import pyqtgraph
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *

# importing system
import sys

# importing numpy as np
import numpy as np

# importing pyqtgraph as pg
import pyqtgraph as pg
from PyQt5.QtGui import *

class VAEScatter(pg.PlotWidget):

    def __init__(self):
        super().__init__()

        self.check = False

        self.setBackground("white")
        self.getPlotItem().hideAxis('bottom')
        self.getPlotItem().hideAxis('left')
        self.scatter = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush(255, 0, 0, 120))

        # getting random position
        n = 300
        pos = np.random.normal(size=(2, n), scale=1e-5)

        self.tempPoints = [{'pos': pos[:, i], 'data': 1}
                 for i in range(n)] + [{'pos': [0, 0], 'data': 1}]
        self.scatter.addPoints(self.tempPoints)
        #self.sceneObj.sigMouseMoved.connect(self.mouseMoveEvent)

        self.scatter.sigHovered.connect(self.mouseMoveEvent)
        self.addItem(self.scatter)

        self.mousePoint = None

    def mousePressEvent(self, ev):
        self.check = True
    def mouseMoveEvent(self, ev):
        if self.check:

            self.scatter.clear()
            self.scatter.addPoints(self.tempPoints)
            boundingRect = self.scatter.boundingRect()
            plot_x = (boundingRect.width()/self.centralWidget.width())*ev.pos().x() + boundingRect.x()
            plot_y = (boundingRect.height()/self.centralWidget.height())*(self.centralWidget.height() - ev.pos().y()) + boundingRect.y()
            self.mousePoint = [{'pos': [plot_x, plot_y], 'data': 0}]
            self.scatter.addPoints(self.mousePoint)
    def mouseReleaseEvent(self, ev):
        self.check = False

        self.scatter.clear()
        self.scatter.addPoints(self.tempPoints)

    def setup(self):
        pass


class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        # setting title
        self.setWindowTitle("VAE Visualizer")

        # setting geometry
        self.setGeometry(100, 100, 1280, 720)


        # calling method
        self.UiComponents()

        # showing all the widgets
        self.show()

    # method for components
    def UiComponents(self):


        self.mainWidget = QWidget()
        self.mainLayout = QHBoxLayout()


        self.vaeScatter = VAEScatter()
        self.imageOverlay = QLabel()
        image = QImage("./myplot.png")
        self.imageOverlay.setPixmap(QPixmap.fromImage(image).scaled(self.imageOverlay.size(), Qt.KeepAspectRatio))
        self.mainLayout.addWidget(self.vaeScatter)
        self.mainLayout.addWidget(self.imageOverlay)

        self.mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.mainWidget)



def main():
    # create pyqt5 app
    App = QApplication(sys.argv)

    # create the instance of our Window
    window = Window()

    # start the app
    sys.exit(App.exec())


if __name__ == "__main__":
    main()
