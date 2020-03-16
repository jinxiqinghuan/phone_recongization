import cv2 as cv
import numpy as np

if __name__ == '__main__':
    src = cv.imread("E:/caller/test/1.jpg")
    print(src.shape)
    hog = cv.HOGDescriptor()
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    fv = hog.compute(gray, winStride=(8, 8), padding=(0, 0))
    print(len(fv))
    print(fv)
    cv.imshow("hog-descriptor", src)
    cv.waitKey(0)
    cv.destroyAllWindows()
