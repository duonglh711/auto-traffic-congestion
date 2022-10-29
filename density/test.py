import cv2
import glob
import numpy as np
import csrnet
import handling
import load_data
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import time

if __name__ == '__main__':

    model = csrnet.CSRNet((720,720,3))
    path = '../weights/dens.h5'
    model = handling.load_model(model, path)
    test = '../data/img/thesis.png'
    img_paths = [test]

    ROI = cv2.resize(cv2.imread('../data/video/ROI.png'), (720,720))
    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)/255.

    for path in range(0,100):
        image = load_data.load_img(test)
        image = cv2.resize(image, (720, 720))
        image[np.where(ROI==0)] = [0,0,0]
        start_time = time.time()
        data = model.predict(np.expand_dims(image, axis=0))
        num = np.sum(data)
        print("FPS: {:.2f}".format(1/(time.time()-start_time)))
        # plt.figure(dpi=300)
        # plt.axis('off')
        # plt.margins(0, 0)
        # plt.imshow(data[0], cmap=CM.jet)
        # plt.savefig('./demo.png', dpi=300, bbox_inches='tight', pad_inches=0)
        # print(num)
    print("Average: ", sum(wrong)/len(wrong) )
