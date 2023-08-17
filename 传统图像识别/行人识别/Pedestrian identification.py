import cv2
import random
import glob
import numpy as np


# 加载本地图像数据集，并将数据集全部添加到列表中，然后打乱数据顺序
def load_image(filename):
    paths = glob.glob(filename)
    persons, labels = [], []
    for i in paths:
        persons.append(cv2.imread(i))
        labels.append(1)
    random.seed(1)
    random.shuffle(persons)
    persons = np.array(persons)
    return persons, labels


# 图像预处理，将输入图像灰度化、高斯模糊
def image_preprocessing(image):
    cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # image = cv2.resize(image, dsize=(32, 64))
    image_preprocess = cv2.GaussianBlur(image, (3, 3), sigmaX=1, sigmaY=1)
    return image_preprocess


# 构建HOG检测器
def get_hog():
    winSize = (64, 128)
    cellSize = (8, 8)
    blockSize = (16, 16)
    blockStride = (16, 16)
    nbins = 9
    signedGradient = True
    derivAperture = 1  # 默认参数
    winSigma = -1.  # 默认参数
    histogramNormType = 0  # 默认参数
    L2HysThreshold = 0.2  # 默认参数
    gammaCorrection = 1  # 默认参数
    nlevels = 64  # 默认参数
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)
    return hog


# 创建SVM模型并配置参数
def SVM_model():
    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_ONE_CLASS)
    model.setKernel(cv2.ml.SVM_POLY)
    model.setC(1)
    model.setNu(0.01)
    model.setDegree(0.1)
    model.setCoef0(0.5)
    model.setGamma(0.6)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, int(1e2), 1e-5))
    return model


# 训练模型
def SVM_train(model, samples, responses):
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model


# 计算分类器准确率
def accuracy(model, data_train, labels_train):
    retval, result = model.predict(data_train)
    temp = (np.array(labels_train) == result).mean()
    print(f'该模型的准确率是：{temp * 100}')


# 测试分类器
def image_predict(model, data_test, samples, labels_test):
    retval, result = model.predict(samples)
    counter = 0
    for i in (labels_test == result.ravel()):
        # 测试结果与实际结果不符合仅呈现红色通道
        if not i:
            data_test[counter][..., :2] = 0
            counter += 1
    h1 = data_test[0]
    for i in data_test[1:12, ...]:
        h1 = np.hstack((h1, i))
    h2 = data_test[12]
    for i in data_test[13:, ...]:
        h2 = np.hstack((h2, i))
    return np.vstack((h1, h2))


if __name__ == "__main__":
    print('加载图片...')
    datas, labels = load_image('image\\Pedestrian detection\\per*.ppm')
    temp, data_test = np.split(datas, [900])

    print('数据预处理...')
    datas = list(map(image_preprocessing, datas))

    print('提取训练数据的HOG特征向量...')
    hog = get_hog()
    hog_vector = list(map(hog.compute, datas))
    # hog_vector = []
    # for i in datas:
    #    hog_vector.append(hog.compute(i))

    print('将数据集分为两部分，900张用于训练，24张用于测试...')
    data_train, temp = np.split(datas, [900])
    labels_train, labels_test = np.split(np.array(labels), [900])
    hog_vector_train, hog_vector_test = np.split(hog_vector, [900])

    print('训练SVM模型...')
    model = SVM_model()
    model_svm = SVM_train(model, hog_vector_train, labels_train)

    print('输出分类模型的准确率...')
    accuracy(model_svm, hog_vector_train, labels_train)

    print('测试分类模型...')
    result = image_predict(model_svm, data_test, hog_vector_test, labels_test)

    cv2.imshow('result, press the q key to exit', result)
    while 1:
        if cv2.waitKey() == ord('q'):
            break

    print('测试其他图像...')
    class_name = {0: "不包含行人", 1: "包含行人"}
    img = cv2.imread('image\\persontrain.png')
    img = cv2.resize(img, dsize=(64, 128))
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_preprocess = cv2.GaussianBlur(img2, (3, 3), sigmaX=1, sigmaY=1)
    # vector = np.array([hog.compute(img_preprocess)])
    vector = np.expand_dims(hog.compute(img_preprocess), 0)
    ret = model_svm.predict(vector)[1].ravel()
    print(f"图片img{class_name[int(ret)]}")

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    