import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from pprint import pprint

def read_data(file_feature, file_label):
    # read a csv file
    df_x = pd.read_csv(file_feature)
    cols = list(df_x.columns)
    # convert pandas dataframe to numpy array
    x = df_x.to_numpy()
    y = pd.read_csv(file_label, header=None).values.astype(int).flatten()

    return x, y, cols


class DecisionTree:
    def __init__(self, feature_names, max_depth):
        self.feature_names = feature_names
        self.depth = 0
        self.max_depth = max_depth


    def entropy(self, y):
        if y.size == 0:
            return 0
        value, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return -(p * np.log(p) / np.log(2)).sum()


    def choose_best_feature(self, x, y):
        best_feature = None
        max_info_gain = 0
        cur_entropy = self.entropy(y)
        m = np.shape(x)[1]
        for i in range(m):
            indices = np.argsort(x[:, i])
            x = x[indices]
            y = y[indices]
            feature_i = x[:, i]

            value, counts = np.unique(feature_i, return_counts=True)
            total = counts.sum()
            if np.shape(counts)[0] == 1:
                if value[0] == 0:
                    num_neg = counts[0]
                    num_pos = 0
                else:
                    num_pos = counts[0]
                    num_neg = 0
            else:
                num_neg = counts[0]
                num_pos = counts[1]

            left_y = y[:num_neg]
            right_y = y[num_neg:]
            left_entropy = self.entropy(left_y)
            right_entropy = self.entropy(right_y)

            cond_entropy = (num_neg / total) * left_entropy + (num_pos/total) * right_entropy
            info_gain = cur_entropy - cond_entropy
            if info_gain >= max_info_gain:
                max_info_gain = info_gain
                best_feature = i

        return best_feature, max_info_gain


    def partition(self, x, y, i):
        feature_i = x[:, i]
        y_left = y[feature_i == 0]
        y_right = y[feature_i == 1]
        return y_left, y_right


    def build_tree(self, x, y, node={}, depth=0):
        #if node is None:
        #    return None

        # no data in this group
        if len(y) == 0:
            return None

        # max depth
        elif depth > self.max_depth:  # base case 4: max depth reached
            return None

        # all same value in y
        elif self.all_same(y):
            return {'label': y[0], 'isLeaf': True}

        else:
            best_feature, gain = self.choose_best_feature(x, y)

            y_left, y_right = self.partition(x, y, best_feature)

            node = {'feature Name': self.feature_names[best_feature],
                        'index': best_feature,
                        'info_gain': gain,
                        'label': int(np.round(np.mean(y))),
                        'isLeaf': False
                    }  # save the information

            # left child: negative
            ans = self.build_tree(x[x[:, best_feature] == 0], y_left, {}, depth + 1)
            if ans == None:
                node['isLeaf'] = True
            else:
                node['left'] = ans

            # right child: positive
            ans = self.build_tree(x[x[:, best_feature] == 1], y_right, {}, depth + 1)
            if ans == None:
                node['isLeaf'] = True
            else:
                node['right'] = ans

            self.trees = node
            return node

    def all_same(self, items):
        return all(x == items[0] for x in items)

    def predict(self, x):
        tree = self.trees
        results = np.array([0] * len(x))
        for i, c in enumerate(x):
            results[i] = self.get_prediction(c)
        return results

    def get_prediction(self, row):
        node = self.trees
        while node.get('isLeaf') == False:
            if row[node['index']] == 0:
                node = node['left']
            else:
                node = node['right']
        else:
            return node.get('label')

    def accuracy(self, y, prediction):
        accuracy = (y == prediction).mean()
        return accuracy


x, y, cols = read_data("pa4_train_X.csv", "pa4_train_y.csv")
x_val, y_val, _ = read_data("pa4_dev_X.csv", "pa4_dev_y.csv")

dmax = [2, 5, 10, 20, 25, 30, 40, 50]
training_acc = []
val_acc = []
for i in range(len(dmax)):
    obj = DecisionTree(feature_names=cols, max_depth=dmax[i])
    decision_tree = obj.build_tree(x, y)
    if dmax[i] <= 2:
        pprint(decision_tree)

    pred_training = obj.predict(x)
    accuracy = obj.accuracy(y, pred_training)
    training_acc.append(accuracy)
    pred_val = obj.predict(x_val)
    accuracy = obj.accuracy(y_val, pred_val)
    val_acc.append(accuracy)


plt.plot(dmax, training_acc, marker='o', label="Training Accuracy")
plt.plot(dmax, val_acc, marker='^', label="Validation Accuracy")
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracies')
plt.grid()
plt.legend()
plt.show()
