# importing Qt widgets
import glob
import os
import pickle

import cv2
import hydra
import pyqtgraph
import qimage2ndarray
import torch
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *

# importing system
import sys

# importing numpy as np
import numpy as np

# importing pyqtgraph as pg
import pyqtgraph as pg
from PyQt5.QtGui import *
from matplotlib import cm

from VAE.models.VAE import VAE


class VAEScatter(pg.PlotWidget):

    def __init__(self):
        super().__init__()

        self.check = False

        self.setBackground("white")
        self.getPlotItem().hideAxis('bottom')
        self.getPlotItem().hideAxis('left')
        self.scatter = pg.ScatterPlotItem()


        self.scatter.sigHovered.connect(self.mouseMoveEvent)
        self.addItem(self.scatter)

        self.mousePointItem = None
        self.mousePoint = None
        self.callback = None


    def initPoints(self,points):

        self.tempPoints = [{'pos': (x, y),
                            'brush': pg.intColor(l)} for (x,y), l in points]
        self.scatter.addPoints(self.tempPoints)
    def setCallback(self, method):
        self.callback = method

    def mousePressEvent(self, ev):
        self.check = True

    def mouseMoveEvent(self, ev):
        if self.check:

            self.scatter.clear()
            self.scatter.addPoints(self.tempPoints)
            boundingRect = self.scatter.boundingRect()
            start_plot_axis_x = boundingRect.x()
            start_plot_axis_y = boundingRect.y()

            plot_width = boundingRect.width()
            plot_height = boundingRect.height()

            screen_plot_width = self.scatter.getViewWidget().size().width() -1
            screen_plot_height = self.scatter.getViewWidget().size().height() -1

            plot_click_x = (plot_width/screen_plot_width) * ev.pos().x() + start_plot_axis_x
            plot_click_y =  (plot_height/screen_plot_height)*(screen_plot_height - ev.pos().y()) + start_plot_axis_y
            self.mousePoint = (plot_click_x, plot_click_y)
            #print(self.mousePoint)
            self.mousePointItem = [{'pos': [plot_click_x, plot_click_y], 'data': 0}]
            self.scatter.addPoints(self.mousePointItem)
            self.callback(self.mousePoint)


    def mouseReleaseEvent(self, ev):
        self.check = False

        self.scatter.clear()
        self.scatter.addPoints(self.tempPoints)

    def setup(self):
        pass

class ModelVAEWrapper():


    def __init__(self, cfg):
        ## Create a model
        # get last checkpoint
        list_of_files = glob.glob(cfg.checkpoint_path + "*")  # * means all if need specific format then *.csv
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
        self.model = VAE.load_from_checkpoint(latest_checkpoint)

        self.latent_space = cfg.latent_space_size

    def generateImage(self, z, size):
        z = torch.Tensor(z)
        z = z.to("cuda:0")
        self.model.to("cuda:0")
        image = self.model.generate(z)
        image = image.to("cpu")
        image = image.squeeze().permute(1, 2, 0).numpy() if image.shape[-3] == 3 else image.squeeze().numpy()
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                      dtype=cv2.CV_32F).astype(np.uint8)
        width, height = size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        return image
class Window(QMainWindow):

    def __init__(self, width, height, cfg):
        super().__init__()

        # setting title
        self.setWindowTitle("VAE Visualizer")
        self.cfg = cfg
        # setting geometry
        self.setGeometry(100, 100, width, height)

        self.model = ModelVAEWrapper(cfg)

        self.prediction = None
        with open(self.cfg.prediction_logdir + "/prediction.pkl", "rb") as f:
            self.prediction = pickle.load(f)

        self.latent_space_size = int(self.prediction[0][1].shape[-1])

        # calling method
        self.UiComponents()
        self.setupPlot(int(self.xInput.text()), int(self.yInput.text()))
        # showing all the widgets
        self.show()



    def setupPlot(self, x_axis, y_axis):


        if not (0 <= x_axis < self.latent_space_size and 0 <= y_axis < self.latent_space_size):
            QMessageBox.information(self, "Invalid Axis Values", "Insert valid axis values between 0 and %s."%str(self.latent_space_size))
            return
        self.vaeScatter.scatter.clear()

        prediction = [((z.squeeze().numpy()[x_axis], z.squeeze().numpy()[y_axis]), label) for image, z, label in self.prediction]
        ls = [l[0] for (x,y), l in prediction]
        labels_idxs = [*map({k: v for v, k in enumerate(dict.fromkeys(ls))}.get, ls)]
        print(labels_idxs)
        points = [ ((x,y), label_indx) for ((x,y), label), label_indx in zip(prediction, labels_idxs)]
        self.vaeScatter.initPoints(points)




    def update_image(self, point):
        random_left_components = [0]*(self.model.latent_space - 2)#np.random.normal(size=(self.model.latent_space - 2))
        z = np.insert(random_left_components, 0, [point[0], point[1]])
        image = self.model.generateImage(z, (self.imageOverlay.size().width(),self.imageOverlay.size().height()))
        q_im = qimage2ndarray.array2qimage(image)
        self.imageOverlay.setPixmap(QPixmap.fromImage(q_im).scaled(self.imageOverlay.size(), Qt.KeepAspectRatio))

    def updatePlot(self):
        try:
            x_axis = int(self.xInput.text())
            y_axis = int(self.yInput.text())
        except:
            return

        self.setupPlot(x_axis, y_axis)
    def UiComponents(self):


        self.mainWidget = QWidget()

        self.mainLayout = QVBoxLayout()

        self.graphics = QWidget()

        self.graficsLayout = QHBoxLayout()
        self.graphics.setLayout(self.graficsLayout)

        self.vaeScatter = VAEScatter()
        self.vaeScatter.setCallback(self.update_image)
        self.imageOverlay = QLabel()
        self.imageOverlay.setFixedWidth(640)
        self.imageOverlay.setFixedHeight(640)

        self.graficsLayout.addWidget(self.vaeScatter)
        self.graficsLayout.addWidget(self.imageOverlay)

        self.AxisControl = QWidget()
        self.axisControlLayout = QHBoxLayout()
        self.axisControlLayout.setContentsMargins(0, 0, 0, 0)
        self.AxisControl.setLayout(self.axisControlLayout)

        self.xLabel = QLabel()
        self.xLabel.setText("X Axis")
        self.xLabel.setAlignment(Qt.AlignLeft)
        self.xLabel.setContentsMargins(0, 0, 0, 0)

        self.yLabel = QLabel()
        self.yLabel.setText("Y Axis")
        self.yLabel.setAlignment(Qt.AlignLeft)

        self.xInput = QLineEdit()
        self.xInput.setFixedWidth(40)
        self.xInput.setText("0")
        self.xInput.setAlignment(Qt.AlignLeft)

        self.yInput = QLineEdit()
        self.yInput.setFixedWidth(40)
        self.yInput.setText("1")
        self.yInput.setAlignment(Qt.AlignLeft)

        self.updateAxisButton = QPushButton()
        self.updateAxisButton.setText("Update Axis")
        self.updateAxisButton.setFixedWidth(90)
        self.updateAxisButton.clicked.connect(self.updatePlot)

        self.axisControlLayout.addWidget(self.xLabel)
        self.axisControlLayout.addWidget(self.xInput)
        self.axisControlLayout.addWidget(self.yLabel)
        self.axisControlLayout.addWidget(self.yInput)
        self.axisControlLayout.addWidget(self.updateAxisButton)

        self.mainLayout.addWidget(self.graphics)
        self.mainLayout.addWidget(self.AxisControl)

        self.mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.mainWidget)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    # create pyqt5 app
    App = QApplication(sys.argv)

    # create the instance of our Window
    window = Window(1280, 640, cfg)
    window.show()
    # start the app
    sys.exit(App.exec())


if __name__ == "__main__":
    main()
