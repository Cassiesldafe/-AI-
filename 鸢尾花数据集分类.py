from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
iris_datasets = load_iris()
X = iris_datasets['data']#特征值
y = iris_datasets['target']#目标值

#数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#print("训练集:\n", X_train, "\n测试集:\n", X_test)
#model
class KNN():
    #初始化，设置neighbors=3,使用L2范数求距离
    def __init__(self, X_train, y_train, n_neighbors=3, ord=2):
        self.x_train = X_train
        self.y_train = y_train
        self.n = n_neighbors
        self.ord = ord
    def predict(self,x):
        #创建一个列表
        knn_list = []
        for i in range(self.n):
            #求距离
            dist = np.linalg.norm(x-self.x_train[i], ord=2)
            #向列表knn_list添加前三个训练样本的距离和label，为元组(dist,label)
            knn_list.append((dist, self.y_train[i]))
        #从第四个样本开始
        for i in range(self.n, len(self.x_train)):
            #找到knn_list中最大值索引
            max_index=knn_list.index(max(knn_list, key=lambda x:x[0]))
            dist = np.linalg.norm(x - self.x_train[i], ord=self.ord)
            #与最大值比较，要把最大值踢出去,就是不断缩小预测样本与训练样本之间的距离
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])
        #统计-看看有没有误判的类别，计算损失
        #knn列表存储了三个label值
        knn = [k[-1] for k in knn_list]
        #用key-value的形式记录label=0或1的1个数有多少个
        count_pairs=Counter(knn)
        #[0:1,1:2],得到最大计数值是label,少数服从多数原则
        max_count=sorted(count_pairs.items(), key=lambda x:x[1])[-1][0]
        return max_count
    def score(self, x_test, y_test):
        right_count = 0
        for x,y in zip(x_test, y_test):
            #调用了KNN.predict
            label = self.predict(x)
            if label == y:
                right_count += 1
        return right_count/len(x_test)

t = KNN(X_train, y_train)

#测试模型
print("测试结果准确率：{:.2f}".format(t.score(X_test, y_test)))

#预测
X_new = np.array([[5, 2.9, 1, 0.2]])
y_new = t.predict(X_new)
print("预测结果：{}".format(y_new))
print("预测类型：{}".format(iris_datasets['target_names'][y_new]))
