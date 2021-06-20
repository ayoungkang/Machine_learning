import pandas as pd
import numpy as np
import csv


class LogisticRegression:
    def read_data(self, file_feature, file_label):
        # read a csv file
        df_x = pd.read_csv(file_feature)
        df_y = pd.read_csv(file_label)

        # move 'dummy' to 1st position
        cols = list(df_x.columns)
        cols = [cols[-1]] + cols[:-1]
        df_x = df_x[cols]

        # convert pandas dataframe to numpy array
        x = df_x.to_numpy()
        y = df_y.to_numpy()
        return x, y, cols

    def normalize(self, vector):
        vec_range = np.ptp(vector)
        if vec_range == 0:
            return vector
        return (vector - np.min(vector)) / vec_range

    def gradient_descent(self, x, y, learning_rate, ld, max_iters, N, M):
        w = np.zeros(M)
        y = y.flatten()
        loss = []

        for i in range(0, max_iters):
            y_hat = self.sigmoid(np.dot(x, w))

            gradient = np.matmul(x.T, y_hat - y) / N
            w = w - learning_rate * gradient

            # L2 norm contribution (Regularization)
            w[1:] -= learning_rate * ld * w[1:]

            cost = obj.cal_loss(y_hat, y, w, ld)
            loss.append(cost)

            if i % 1000 == 0:
                print(f"iteration({i}) loss: {cost}")

        print("number of iterations:", i + 1)
        return w, loss[-1]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def cal_loss(self, h, y, w, ld):
        n = h.shape[0]
        return np.sum((-y * np.log(h) - (1 - y) * np.log(1 - h))) \
               / n + ld * np.linalg.norm(w[1:]) ** 2

    def predict(self, x, w):
        return np.round(self.sigmoid(np.dot(x, w)))x




obj = LogisticRegression()
x, y, col_names = obj.read_data("pa2_train_X.csv", "pa2_train_y.csv")
# normalize numeric and ordinal features (age, annual premium, vintage)
numeric_features = [2, 6, 7]
for i in numeric_features:
    x[:, i] = obj.normalize(x[:, i])

max_iters = 10000
learning_rate = 0.1
ld = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

# N: num of training examples, M: num of features
N, M = np.shape(x)

w_mat = []
accuracy_mat = []
loss_mat = []
#w = np.random.normal(0.0, 1.0, size=(M,))
for i in range(len(ld)):
    w, loss = obj.gradient_descent(x, y, learning_rate, ld[i], max_iters, N, M)
    y_preds = obj.predict(x, w)
    w_mat.append(w)
    loss_mat.append(loss)
    accuracy = (y.flatten() == y_preds).mean()
    accuracy_mat.append(accuracy)


print("Training Accuracy:", accuracy_mat)
print("Loss:", loss_mat)

with open('w_mat.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerow(col_names)
    write.writerows(w_mat)

with open('accuracy.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerow(accuracy_mat)

x_dev, y_dev, col_names = obj.read_data("pa2_dev_X.csv", "pa2_dev_y.csv")
accuracy_dev_mat = []
# normalize numeric and ordinal features
for i in numeric_features:
    x_dev[:, i] = obj.normalize(x_dev[:, i])

for i in range(len(w_mat)):
    y_dev_preds = obj.predict(x_dev, w_mat[i])
    accuracy = (y_dev.flatten() == y_dev_preds).mean()
    accuracy_dev_mat.append(accuracy)

print("Validation Accuracy:", accuracy_dev_mat)

with open('accuracy_dev.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerow(accuracy_dev_mat)

zeros_cnt = []
for i in range(len(w_mat)):
    w = w_mat[i]
    cnt = len(w) - np.count_nonzero(w)
    zeros_cnt.append(cnt)

print("Sparsity", zeros_cnt)
