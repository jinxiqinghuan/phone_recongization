import cv2 as cv
import os
import numpy as np

def get_hog_descriptor(image):
    # https://pastebin.com/Y1LXaRrE
    hog = cv.HOGDescriptor()
    h, w = image.shape[:2]
    rate = 64 / w
    image = cv.resize(image, (64, np.int(rate*h)))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    bg = np.zeros((128, 64), dtype=np.uint8)
    bg[:,:] = 127
    h, w = gray.shape
    dy = (128 - h) // 2
    bg[dy:h+dy,:] = gray
    cv.imshow("hog_bg", bg)
    cv.waitKey(0)
    # 64x128 = 3780
    fv = hog.compute(bg, winStride=(8, 8), padding=(0, 0))
    return fv

def generate_dataset(pdir, ndir):
    train_data = []
    labels = []
    for file_name in os.listdir(pdir):
        img_dir = os.path.join(pdir, file_name)
        img = cv.imread(img_dir)
        hog_desc = get_hog_descriptor(img)
        one_fv = np.zeros([len(hog_desc)], dtype=np.float32)
        for i in range(len(hog_desc)):
            one_fv[i] = hog_desc[i][0]
        train_data.append(one_fv)
        labels.append(1)

    for file_name in os.listdir(ndir):
        img_dir = os.path.join(ndir, file_name)
        img = cv.imread(img_dir)
        hog_desc = get_hog_descriptor(img)
        one_fv = np.zeros([len(hog_desc)], dtype=np.float32)
        for i in range(len(hog_desc)):
            one_fv[i] = hog_desc[i][0]
        train_data.append(one_fv)
        labels.append(-1)
    return np.array(train_data, dtype=np.float32), np.array(labels, dtype=np.int32)

def svm_train(positive_dir, negative_dir):
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(25.383)
    trainData, responses = generate_dataset(positive_dir, negative_dir)
    responses = np.reshape(responses, [-1, 1])
    svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
    svm.save('svm_data.dat')
def elec_detect(image):
    hog_desc = get_hog_descriptor(test_img)
    print(len(hog_desc))
    one_fv = np.zeros([len(hog_desc)], dtype=np.float32)
    for i in range(len(hog_desc)):
        one_fv[i] = hog_desc[i][0]
    one_fv = np.reshape(one_fv, [-1, len(hog_desc)])
    print(len(one_fv), len(one_fv[0]))
    svm = cv.ml.SVM_load('svm_data.dat')
    result = svm.predict(one_fv)[1]
    print('是否在打电话：1表示“是”，-1表示“不是”')
    print(result)

if __name__ == '__main__':
   # generate_dataset("E:/caller/positive/", "E:/caller/negative/")
   
    #svm_train("E:/caller/positive/", "E:/caller/negative/")
    
    test_img = cv.imread("E:/caller/test/4.jpg")
    cv.waitKey(0)
    elec_detect(test_img)
    cv.destroyAllWindows()
    

