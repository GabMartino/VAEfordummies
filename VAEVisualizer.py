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
            plot_x = (boundingRect.width()/self.centralWidget.width())*ev.pos().x() + boundingRect.x()
            plot_y = (boundingRect.height()/self.centralWidget.height())*(self.centralWidget.height() - ev.pos().y()) + boundingRect.y()
            self.mousePoint = (plot_x, plot_y)
            self.mousePointItem = [{'pos': [plot_x, plot_y], 'data': 0}]
            self.scatter.addPoints(self.mousePointItem)
            self.callback(self.mousePoint)


    def mouseReleaseEvent(self, ev):
        self.check = False

        self.scatter.clear()
        self.scatter.addPoints(self.tempPoints)

    def setup(self):
        pass

class ModelVAEWrapper():


    def __init__(self, cfg,):
        ## Create a model
        self.model = VAE(cfg)
        # get last checkpoint
        list_of_files = glob.glob(cfg.checkpoint_path + "*")  # * means all if need specific format then *.csv
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
        self.model = self.model.load_from_checkpoint(latest_checkpoint)

        self.latent_space = cfg.latent_space_size

    def generateImage(self, z):
        z = torch.Tensor(z)
        image = self.model.generate(z)
        image = cv2.normalize(image.squeeze().permute(1, 2, 0).numpy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                      dtype=cv2.CV_32F).astype(np.uint8)
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

        # calling method
        self.UiComponents()
        self.setupPlot()
        # showing all the widgets
        self.show()



    def setupPlot(self):
        prediction = None
        with open(self.cfg.prediction_logdir + "/prediction.pkl", "rb") as f:
            prediction = pickle.load(f)

        prediction = [((z.squeeze().numpy()[0], z.squeeze().numpy()[1]), label) for image, z, label in prediction]
        ls = [l[0] for (x,y), l in prediction]
        labels_idxs = [*map({k: v for v, k in enumerate(dict.fromkeys(ls))}.get, ls)]

        points = [ ((x,y), label_indx) for ((x,y), label), label_indx in zip(prediction, labels_idxs)]
        self.vaeScatter.initPoints(points)



    def update_image(self, point):
        random_left_components = np.random.normal(size=(self.model.latent_space - 2))
        z = np.insert(random_left_components, 0, [point[0], point[1]])
        image = self.model.generateImage(z)
        print(image.shape)
        q_im = qimage2ndarray.array2qimage(image)
        self.imageOverlay.setPixmap(QPixmap.fromImage(q_im).scaled(self.imageOverlay.size(), Qt.KeepAspectRatio))

    def UiComponents(self):


        self.mainWidget = QWidget()
        self.mainLayout = QHBoxLayout()


        self.vaeScatter = VAEScatter()
        self.vaeScatter.setCallback(self.update_image)
        self.imageOverlay = QLabel()
        self.imageOverlay.setFixedWidth(640)
        self.imageOverlay.setFixedHeight(640)

        self.mainLayout.addWidget(self.vaeScatter)
        self.mainLayout.addWidget(self.imageOverlay)

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
