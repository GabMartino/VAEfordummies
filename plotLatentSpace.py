import pickle

import hydra
import matplotlib.pyplot as plt


import cv2
import numpy as np
from matplotlib import cm

def showMultipleImages(images, w= 10, h = 10):

    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = 5
    for i in range(1, columns*rows + 1):
        img = images[i - 1]
        print(img)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    # Fetch predictions



    prediction = None
    with open(cfg.prediction_logdir + "/prediction.pkl", "rb") as f:
        prediction = pickle.load(f)

    prediction = [ (cv2.normalize(image.squeeze().permute(1,2,0).numpy(), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8), z, label) for image, z, label in prediction]

    images = [i for i, z, l in prediction]
    zs = np.array([np.array(z) for i, z, l in prediction]).squeeze()
    ls = [l[0] for i, z, l in prediction]
    labels_idxs = [*map({k: v for v, k in enumerate(dict.fromkeys(ls))}.get, ls)]
    num_labels = np.unique(ls).size
    colorMap = cm.get_cmap('viridis', num_labels)
    plt.scatter(zs[:, 0], zs[:, 1], c=labels_idxs, cmap=colorMap)

    plt.show()
    plt.scatter(zs[:, 1], zs[:, 2], c=labels_idxs, cmap=colorMap)
    plt.show()
    plt.scatter(zs[:, 2], zs[:, 0], c=labels_idxs, cmap=colorMap)
    plt.show()
    plt.scatter(zs[:, 0], zs[:, 3], c=labels_idxs, cmap=colorMap)
    plt.show()
    plt.plot(np.mean(zs, axis=0))
    plt.show()

    showMultipleImages(images[:25])
    plt.imshow(prediction[0][0])
    plt.show()
if __name__ == "__main__":

    main()