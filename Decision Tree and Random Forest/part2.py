import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
np.random.seed(1)


def read_data(file_feature, file_label):
    # read a csv file
    df_x = pd.read_csv(file_feature)
    cols = list(df_x.columns)
    # convert pandas dataframe to numpy array
    x = df_x.to_numpy()
    y = pd.read_csv(file_label, header=None).values.astype(int).flatten()

    return x, y, cols

class RandomForest:
    def __init__(self, feature_names, max_depth):
        self.feature_names = feature_names
        self.depth = 0
        self.max_depth = max_depth

    def draw_bootstrap(self, x, y):
        bootstrap_indices = np.random.choice(range(len(x)), len(x), replace=True)
        x_bootstrap = x[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]
        return x_bootstrap, y_bootstrap


    def entropy(self, y):
        if y.size == 0:
            return 0
        value, counts = np.unique(y, return_counts=True)
        #print(value, counts)
        p = counts / counts.sum()
        return -(p * np.log(p) / np.log(2)).sum()




    def choose_best_feature(self, x_bootstrap, y_bootstrap, m_features):
        best_feature = None
        max_info_gain = 0

        # m: number of features
        m = np.shape(x_bootstrap)[1]
        # randomly sampled features
        #features = np.random.sample(range(m), m_features)
        #features = list(np.random.permutation(np.arange(0, m))[:m_features])
        features = list(np.random.choice(range(m), m_features, replace=False))
        #print(features)
        x_bootstrap = x_bootstrap[:, features]


        cur_entropy = self.entropy(y_bootstrap)

        for i in range(m_features):
            #print("i", i)
            indices = np.argsort(x_bootstrap[:, i])
            x_bootstrap = x_bootstrap[indices]
            y_bootstrap = y_bootstrap[indices]
            feature_i = x_bootstrap[:, i]
            #print("feature", feature_i)
            #print("y", y)

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

            left_y = y_bootstrap[:num_neg]
            right_y = y_bootstrap[num_neg:]
            left_entropy = self.entropy(left_y)
            right_entropy = self.entropy(right_y)

            cond_entropy = (num_neg / total) * left_entropy + (num_pos/total) * right_entropy
            #print(cond_entropy)
            info_gain = cur_entropy - cond_entropy
            if info_gain >= max_info_gain:
                max_info_gain = info_gain
                best_feature = features[i]
        #print("best_feature:", best_feature)
        return best_feature, max_info_gain



    def partition(self, x, y, i):
        feature_i = x[:, i]
        # left = negative, right = positive
        #x_left = x[feature_i == 0]
        y_left = y[feature_i == 0]
        #x_right = x[feature_i == 1]
        y_right = y[feature_i == 1]
        return y_left, y_right


    def build_tree(self, x_bootstrap, y_bootstrap, m, node={}, depth=0):

        if node is None:
            return None

        # no data in this group
        elif len(y_bootstrap) == 0:
            return None

        # max depth
        elif depth > self.max_depth:
            return None

        # all same value in y
        elif self.all_same(y_bootstrap):  # base case 3: all y is the same in this group
            return {'label': y_bootstrap[0], 'isLeaf': True}

        else:
            best_feature, gain = self.choose_best_feature(x_bootstrap, y_bootstrap, m)
            #print(best_feature, gain)
            #print("size:", np.shape(x_bootstrap))
            y_left, y_right = self.partition(x_bootstrap, y_bootstrap, best_feature)

            node = {'feature Name': self.feature_names[best_feature],
                    'index': best_feature,
                    'info_gain': gain,
                    'label': int(np.round(np.mean(y_bootstrap))),
                    'isLeaf': False
                    }


            ans = self.build_tree(x_bootstrap[x_bootstrap[:, best_feature] == 0], y_left, m, {}, depth + 1)
            if ans == None:
                node['isLeaf'] = True
            else:
                node['left'] = ans

            ans = self.build_tree(x_bootstrap[x_bootstrap[:, best_feature] == 1], y_right, m, {}, depth + 1)
            if ans == None:
                node['isLeaf'] = True
            else:
                node['right'] = ans

            self.depth += 1
            #self.trees = node
            return node

    def all_same(self, items):
        return all(x == items[0] for x in items)

    def predict(self, x):
        #trees = self.trees
        results = np.array([0] * len(x))
        for i, c in enumerate(x):
            results[i] = self._get_prediction(c)
        return results

    def _get_prediction(self, row):
        trees = self.subtrees
        vote = []
        for node in trees:
            while node.get('isLeaf') == False:
                #print("index:", node['index'])
                #print("value:", row[node['index']])
                if row[node['index']] == 0:
                    #print("left:", node['left'])
                    node = node['left']
                else:
                    #print("right", node['right'])
                    node = node['right']
            else:
                #return node.get('label')
                #print("label:", node.get('label'))
                vote.append(node.get('label'))
        #print(vote)
        return int(np.round(np.mean(vote)))
        #return max(set(vote), key=vote.count)

    def random_forest(self, x, y, T, m):
        sub_trees = []
        for i in range(T):
            x_bootstrap, y_bootstrap = self.draw_bootstrap(x, y)
            tree = self.build_tree(x_bootstrap, y_bootstrap, m)

            sub_trees.append(tree)
            self.subtrees = sub_trees

        return sub_trees


    def accuracy(self, y, prediction):
        accuracy = (y == prediction).mean()
        return accuracy



x, y, cols = read_data("pa4_train_X.csv", "pa4_train_y.csv")
x_val, y_val, _ = read_data("pa4_dev_X.csv", "pa4_dev_y.csv")


dmax = [2, 10, 25]
m = [5, 25, 50, 100]
T = [10*i for i in range(1,11)]
training_acc = []
val_acc = []

obj = RandomForest(feature_names=cols, max_depth=dmax[1])
for i in range(len(m)):
    training_acc_ = []
    val_acc_ = []
    for j in range(len(T)):
        trees = obj.random_forest(x, y, T[j], m[i])
        #for tree in trees:
        #    print(tree)
        pred_training = obj.predict(x)
        accuracy = obj.accuracy(y, pred_training)
        training_acc_.append(accuracy)
        pred_val = obj.predict(x_val)
        accuracy_val = obj.accuracy(y_val, pred_val)
        val_acc_.append(accuracy_val)

    training_acc.append(training_acc_)
    val_acc.append(val_acc_)


    plt.plot(T, training_acc_, marker='o', label="m = %s" % m[i])
    plt.xlabel('Ensemble Size (T)')
    plt.ylabel('Accuracy')

    title = 'Accuracy of Random Forest (Maximum Depth = {})'.format(dmax[1])
    plt.title(title)
    plt.legend()
plt.grid()
plt.show()

print(training_acc)
print(val_acc)

with open('training_accuracy.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerows(training_acc)

with open('validation_accuracy.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerows(val_acc)
