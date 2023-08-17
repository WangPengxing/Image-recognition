import cv2
import numpy as np
# 本地模块common
from common import clock, mosaic

SZ = 20
CLASS_N = 10


def split2d(img, cell_size, flatten=True):
    """
    将图像按照cell_size尺寸分割，将分割后的数组放置在一个新数组中并返回
    :param img: 将要被分割的图像
    :param cell_size: 分割尺寸
    :param flatten: 是否扁平化从的标志
    :return: img分割后的图像数组集
    """
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = []
    # 按行切分输入的图像
    for row in np.vsplit(img, h//sy):
        cells.append(np.hsplit(row, w // sx))
    # 将列表转换称数组，shape=(h//sy, w//sx, sy, sx)
    cells = np.array(cells)
    if flatten:
        # 将数组cells转为三维数组，shape=((h//sy)*(w//sx), sy, sx),(h//sy)*(w//sx)是切割后的图像数量
        cells = cells.reshape((-1, sy, sx))
    return cells


# 建立训练资源库和标签组
def load_digits(fn):
    """
    建立训练资源库和对应的标签组
    :param fn:输入图片的位置
    :return:分割后的图像集合以及对应的标签组
    """
    digits_img = cv2.imread(fn, 0)
    # 通过split2d()函数分割读取的图像
    digits = split2d(digits_img, (SZ, SZ))
    # 建立[0, 9]的整数类，对应手写数字，数量与len(digits)同
    labels = np.repeat(np.arange(CLASS_N), len(digits) / CLASS_N)
    return digits, labels


# 纠偏
def deskew(img):
    """
    判断图像倾斜是否在[0,1e-2)之内，若在返回一个副本，否则返回一个经过错切的图像
    :param img: 输入图像
    :return: 返回一个在倾斜角度内的图像
    """
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    # 创建错切矩阵
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    # 错切变换
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def svmInit(C=12.5, gamma=0.50625):
    """
    创建并配置SVM模型
    :param C: 参数C
    :param gamma: 参数Gamma
    :return: 返回一个已经配置完参数的SVM模型
    """
    # 创建一个空SVM模型
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(C)
    # SVM核为SVM_RBF，需要有对应的Gamma参数
    model.setKernel(cv2.ml.SVM_RBF)
    # SVM类型为SVM_C_SVC，需要有对应的C参数
    model.setType(cv2.ml.SVM_C_SVC)

    return model


def svmTrain(model, samples, responses):
    """
    返回经过训练的模型
    :param model: 将要被训练的模型
    :param samples: 训练数据
    :param responses: 数据对应的标签
    :return: 返回经过训练的模型
    """
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model


def svmPredict(model, samples):
    """
    返回模型对测试数据的预测结果
    :param model: 已经训练好的模型
    :param samples: 测试用的图像的特征向量
    :return: 返回模型对测试数据的预测结果
    """
    return model.predict(samples)[1].ravel()


def svmEvaluate(model, digits, samples, labels):
    """
    返回一个打乱了的测试数据集，其中预测错误的数据是红色的
    :param model: 已经训练好的模型
    :param digits: 测试用的数据
    :param samples: 测试用的数据的特征向量
    :param labels: 测试用的数据对应的标签
    :return: 返回一个打乱了的测试数据集，其中预测错误的数据是红色的
    """
    predictions = svmPredict(model, samples)
    accuracy = (labels == predictions).mean()
    print('预测准确率: %.2f %%' % (accuracy * 100))

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, predictions):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)
    print(sum(sum(confusion)))

    vis = []
    for img, flag in zip(digits, predictions == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[..., :2] = 0
        vis.append(img)
    return mosaic(25, vis)


# 图像归一化
def preprocess_simple(digits):
    return np.float32(digits).reshape(-1, SZ * SZ) / 255.0


def get_hog():
    """
    :return: 返回一个获得HOG描述符的函数
    """
    winSize = (20, 20)
    blockSize = (8, 8)
    blockStride = (4, 4)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)

    return hog


if __name__ == '__main__':

    print('加载本地图片 ... ')
    # 本地数据
    digits, labels = load_digits("image\\digits.png")

    print('打乱数据 ... ')
    # 打乱数据
    rand = np.random.RandomState(10)
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]

    print('纠偏 ... ')
    # 纠偏
    digits_deskewed = list(map(deskew, digits))

    print('定义HOG参数 ...')
    # 定义HOG参数
    hog = get_hog()

    print('提取每张图片的HOG特征向量 ... ')
    hog_descriptors = []
    # 提取提取每张图像的特征向量，并添加到列表hog_descriptors
    for img in digits_deskewed:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)

    print('将数据集分成两份，90%用于训练，10%用于测试... ')
    train_n = int(0.9 * len(hog_descriptors))
    digits_train, digits_test = np.split(digits_deskewed, [train_n])
    hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])

    print('训练SVM模型 ...')
    model = svmInit()
    svmTrain(model, hog_descriptors_train, labels_train)

    print('Evaluating model ... ')
    vis = svmEvaluate(model, digits_test, hog_descriptors_test, labels_test)

    cv2.imwrite("digits-classification.jpg", vis)
    cv2.imshow("Vis", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    