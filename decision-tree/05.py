import numpy as np

def calcGini(feature, label, index):
    '''
    计算基尼系数
    :param feature:测试用例中字典里的feature，类型为ndarray
    :param label:测试用例中字典里的label，类型为ndarray
    :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
    :return:基尼系数，类型float
    '''

    #********* Begin *********#
    def GINI(DV,label):
        K=np.unique(label)
        S=len(label)
        temp=0
        for k in K:
            pk=sum(label==k)/S
            temp+=pk*pk
        return 1-temp
    V=np.unique(feature[:,index])
    D=len(feature)
    temp=0
    for v in V:
        Dv=sum(feature[:,index]==v)
        DV=(feature[:,index]==v)
        temp+=(Dv/D)*GINI(DV,label[DV])
    return temp
    #********* End *********#