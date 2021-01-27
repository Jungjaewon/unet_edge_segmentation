import cv2
import numpy as np

if __name__ == '__main__':


    sketch_path = './1020_sketch.png'

    img = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)

    shape = np.shape(img)
    print(np.shape(img))
    print(img)

    print(np.where(img > 128, 1, 0))