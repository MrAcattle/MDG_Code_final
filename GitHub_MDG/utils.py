import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import itertools


def read_data(error=0, unit=[0], is_train=True, addAbs=True):
    """
    Args:
        error (int): The index of error, 0 means normal data
        is_train (bool): Read train or test data
        unit: 一个列表，用来选择需要的某些特定观察变量
    Returns:
        data
    """
    if is_train: # 是否是训练数据 suffix：后缀
        suffix = '.dat'
    else:
        suffix = '_te.dat'
    fi = 'data/d{:02d}{}'.format(error, suffix)
    # 第一个{02d}表示error为两个宽度的十进制数显示（也可以直接当成故障标签）
    # 第二个{ }表示用字符串显示suffix

    data = np.fromfile(fi, dtype=np.float32, sep='   ')

    # data = np.loadtxt(fi)

    if fi == 'data/d00.dat':
        data = data.reshape(-1, 500).T
        data = data[:, unit]

        # 增加抽象节点（30）
        if addAbs:
            abs_nodes = np.zeros((data.shape[0], 30))  # shape[0]是行数
            data = np.concatenate((data, abs_nodes), axis=1)

        li = np.array([0 for i in range(500)]).reshape(500, -1)
        data = np.concatenate((data, li), axis=1)  # axis = 1 表示水平拼接

    #  所有te的数据在else里处理
    else:
        data = data.reshape(-1, 52)
        data = data[:, unit]

        # 增加抽象节点（30）
        if addAbs:
            abs_nodes = np.zeros((data.shape[0], 30))  # shape[0]是行数
            data = np.concatenate((data, abs_nodes), axis=1)

        # 做标记
        li = np.array([0 for i in range(160)])  # 前160是正常数据
        li_f = np.array([error for i in range(800)])  # 后800是故障数据
        temp = np.concatenate((li, li_f), axis=0)
        temp = temp.reshape(960, -1)
        data = np.concatenate((data, temp), axis=1)  # axis = 1 表示水平拼接

    return data


def get_train_data(unit):
    train_data = read_data(error=0, unit=unit, is_train=True)
    train_data = preprocessing.StandardScaler().fit_transform(train_data)  # 归一化
    # 做标记  因为涉及到归一化，所以标签要放在归一化之后 !!!
    li = np.array([0 for i in range(500)]).reshape(500, -1)
    train_data = np.concatenate((train_data, li), axis=1)  # axis = 1 表示水平拼接
    return train_data


def get_test_data(unit):
    test_data = []
    for i in range(22):
        data = read_data(error=i, unit=unit, is_train=False)
        test_data.append(data)
    test_data = np.concatenate(test_data)  # 矩阵拼接

    label = test_data[:, 52]  # 保存标记
    label = np.array(label).reshape(21120, -1)  # test_data.shape[0]

    train_data = read_data(error=0, unit=unit, is_train=True)
    scaler = preprocessing.StandardScaler().fit(train_data)  # 用训练数据归一化
    test_data = scaler.transform(test_data[:, 0:52])
    test_data = np.concatenate((test_data, label), axis=1)

    return test_data


def get_whole_data(unit):
    train_data = get_train_data(unit)
    test_data = get_test_data(unit)
    whole_data = np.concatenate((train_data, test_data))
    return whole_data

def newReadData(unit, addAbsNode=True):
    allData = read_data(error=0, unit=unit, is_train=True, addAbs=addAbsNode)  # 把d00载入
    for i in range(1, 22):
        data = read_data(error=i, unit=unit, is_train=False, addAbs=addAbsNode)
        allData = np.concatenate((allData, data), axis=0)
    dataMatrix = allData[:, :-1]
    # 归一化
    scaler = preprocessing.StandardScaler().fit(dataMatrix[:500, :])
    dataMatrix[:500, :] = preprocessing.StandardScaler().fit_transform(dataMatrix[:500, :])
    dataMatrix[500:, :] = scaler.transform(dataMatrix[500:, :])
    labels = allData[:, -1]
    return dataMatrix, labels

def spit_data(unit):
    train_data = read_data(error=0, unit=unit, is_train=True)  # 把d00载入
    #
    normal_data = []
    fault_data1 = []
    fault_data2 = []
    for i in range(1, 22):
        data = read_data(error=i, unit=unit, is_train=False)
        # 提取normal部分
        normal_data.append(data[0:160, :])
        # 提取fault1部分
        fault_data1.append(data[160:760, :])
        # 提取fault2部分
        fault_data2.append(data[760:960, :])

    # normal的ndarray
    normal_data = np.concatenate(normal_data)
    normal_data = np.concatenate((train_data, normal_data))
    # fault的ndarray
    fault_data1 = np.concatenate(fault_data1)
    fault_data2 = np.concatenate(fault_data2)

    # 归一化 用normal归一化fault
    label_normal = normal_data[:, 52]  # 保存标记
    label_normal = np.array(label_normal).reshape(normal_data.shape[0], -1)
    scaler = preprocessing.StandardScaler().fit(normal_data[:, 0:52])
    normal_data = preprocessing.StandardScaler().fit_transform(normal_data[:, 0:52])
    normal_data = np.concatenate((normal_data, label_normal), axis=1)

    label_fault = fault_data1[:, 52]  # 保存标记
    label_fault = np.array(label_fault).reshape(fault_data1.shape[0], -1)  # test_data.shape[0]
    fault_data1 = scaler.transform(fault_data1[:, 0:52])
    fault_data1 = np.concatenate((fault_data1, label_fault), axis=1)

    label_fault = fault_data2[:, 52]  # 保存标记
    label_fault = np.array(label_fault).reshape(fault_data2.shape[0], -1)  # test_data.shape[0]
    fault_data2 = scaler.transform(fault_data2[:, 0:52])
    fault_data2 = np.concatenate((fault_data2, label_fault), axis=1)

    # 以 3 比 1 比例构建测试集和训练集
    train_data = np.concatenate((normal_data[0:2880, :], fault_data1))
    test_data = np.concatenate((normal_data[2880:3860, :], fault_data2))
    # whole_data = np.concatenate((train_data, test_data))
    return train_data, test_data


def add_abs_node(data_tr, data_te, num_abs):
    node_features_tr = data_tr[:, 0:52]  # label = train_Data[:, 52]
    node_features_te = data_te[:, 0:52]
    # 添加抽象节点特征（30个抽象节点）
    abs_nodes = np.zeros((node_features_tr.shape[0], num_abs))  # shape[0]是行数
    node_features_tr = np.concatenate((node_features_tr, abs_nodes), axis=1)
    abs_nodes = np.zeros((node_features_te.shape[0], num_abs))
    node_features_te = np.concatenate((node_features_te, abs_nodes), axis=1)
    return node_features_tr, node_features_te

def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        ac_sum = 0
        for i in range(cm.shape[0]):
            print("class_ac:", i, format(cm[i][i], '.3f'))
            ac_sum += cm[i][i]
        ac_sum = ac_sum / cm.shape[0]
        print("weighted_ac:", format(ac_sum, '.3f'))
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()





