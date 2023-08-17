import cv2
import numpy as np
import random

# 设置显示框
width = 512
height = 512
win = np.zeros((height, width, 3), dtype=np.uint8)


def training_data(training_sample, sep):
    global win
    # 自定义数据集
    # 每个类别的训练样本数量
    training_samples = training_sample
    # 线性可分的样本比例
    frac_linear_sep = sep

    # 设置训练样本和标签
    train_data = np.empty((training_samples * 2, 2), dtype=np.float32)
    labels = np.empty((training_samples * 2, 1), dtype=np.int32)

    # 随机数种子
    random.seed(100)
    # 线性可分的样板训练数量
    linear_samples = int(frac_linear_sep * training_samples)

    # 创建线性可分class1，class1内的点坐标x∈[0, 0.4*width), y∈[0, 512)
    train_class = train_data[0:linear_samples, :]
    # 定义点的x坐标
    c = train_class[:, 0:1]
    c[:] = np.random.uniform(0.0, 0.4 * width, c.shape)
    # 定义点的y坐标
    c = train_class[:, 1:2]
    c[:] = np.random.uniform(0.0, height, c.shape)

    # 创建线性可分class2，class2内的点坐标x∈[0.6, width), y∈[0, 512)
    train_class = train_data[(training_samples * 2 - linear_samples):training_samples * 2, :]
    # 定义点的x坐标
    c = train_class[:, 0:1]
    c[:] = np.random.uniform(0.6 * width, width, c.shape)
    # 定义点的y坐标
    c = train_class[:, 1:2]
    c[:] = np.random.uniform(0.0, height, c.shape)

    # 创建线性不可分class，class内的点坐标x∈[0.4*width, 0.6*width), y∈[0, 512)
    train_class = train_data[linear_samples:(training_samples * 2 - linear_samples), :]
    # 定义点的x坐标
    c = train_class[:, 0:1]
    c[:] = np.random.uniform(0.4 * width, 0.6 * width, c.shape)
    # 定义点的y坐标
    c = train_class[:, 1:2]
    c[:] = np.random.uniform(0.0, height, c.shape)

    # 创建class1与class2的标签
    labels[0:training_samples, :] = 1
    labels[training_samples:training_samples * 2, :] = 2

    return train_data, labels


def SVM_Model():
    # 创建一个参数已经定义的模型model
    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC)
    model.setC(0.1)
    model.setKernel(cv2.ml.SVM_LINEAR)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, int(1e7), 1e-6))
    return model


def train_model(model, samples, layout, reponses):
    # 训练SVM模型
    model.train(samples, layout, reponses)
    return model


def display_area(model, color1, color2):
    global win
    # 显示决策区域
    for i in range(win.shape[0]):
        for j in range(win.shape[1]):
            sample_mat = np.matrix([[j, i]], dtype=np.float32)
            response = model.predict(sample_mat)[1]
            if response == 1:
                win[i, j] = color1
            else:
                win[i, j] = color2


def display_class(model, train_data, color1, color2, training_samples):
    global win
    # 按类显示训练数据中的点
    for i in range(training_samples):
        px = train_data[i, 0]
        py = train_data[i, 1]
        cv2.circle(win, (int(px), int(py)), 3, color1, -1)
    for i in range(training_samples, 2*training_samples):
        px = train_data[i, 0]
        py = train_data[i, 1]
        cv2.circle(win, (int(px), int(py)), 3, color2, -1)
    sv = model.getUncompressedSupportVectors()
    for i in range(sv.shape[0]):
        cv2.circle(win, (int(sv[i, 0]), int(sv[i, 1])), 6, (200, 200, 200), 2)


if __name__ == '__main__':
    training_samples = 150
    frac_linear_sep = 0.9

    print('创建训练数据'.center(50, '-'))
    train_data, labels = training_data(training_samples, frac_linear_sep)
    print('创建SVM模型'.center(50, '-'))
    svm_model = SVM_Model()
    print('训练SVM模型'.center(50, '-'))
    svm_model = train_model(svm_model, train_data, cv2.ml.ROW_SAMPLE, labels)
    print('显示决策区域'.center(50, '-'))
    display_area(svm_model, (112, 125, 247), (168, 159, 121))
    print('按类显示训练数据中的点'.center(50, '-'))
    display_class(svm_model, train_data, (52, 70, 244), (107, 100, 69), training_samples)

    cv2.imshow('111', win)
    while 1:
        if cv2.waitKey(0) == ord('q'):
            break
    cv2.destroyAllWindows()
