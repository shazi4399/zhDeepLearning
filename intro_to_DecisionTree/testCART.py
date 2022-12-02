# encoding = utf-8


def CTree():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    iris = load_iris()
    # 获取特征集和分类label
    features = iris.data
    labels = iris.target
    # 随机抽取33%的数据作为测试集，其余为训练集
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size = 0.33, random_state= 0)
    # 创建CART分类树
    clf = DecisionTreeClassifier(criterion='gini')
    # 拟合构造CART分类树
    clf = clf.fit(train_features, train_labels)
    # 用CART分类做预测
    test_predict = clf.predict(test_features)
    # 预测结果与测试结果作对比
    score = accuracy_score(test_labels, test_predict)
    print("CART分类树准确率 %.4lf" % score)

def RTree():
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_boston
    from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
    from sklearn.tree import DecisionTreeRegressor
    #准备数据集
    # 准备数据集
    boston = load_boston()
    # 探索数据
    print(boston.feature_names)
    # 获取特征集和房价
    features = boston.data
    prices = boston.target
    # 随机抽取33%的数据作为测试集，其余为训练集
    train_features, test_features, train_price, test_price = \
        train_test_split(features, prices, test_size=0.33)
    # 创建CART回归树
    dtr = DecisionTreeRegressor()
    # 拟合构造CART回归树
    dtr.fit(train_features, train_price)
    # 预测测试集中的房价
    predict_price = dtr.predict(test_features)
    # 测试集的结果评价
    print('回归树二乘偏差均值:', mean_squared_error(test_price, predict_price))
    print('回归树绝对值偏差均值:', mean_absolute_error(test_price, predict_price))



if __name__ == "__main__":
    CTree()
    RTree()

