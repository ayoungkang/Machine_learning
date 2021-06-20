import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt


class Perceptron:
    def read_data(self, file_feature, file_label):
        # read a csv file
        df_x = pd.read_csv(file_feature)
        df_y = pd.read_csv(file_label)

        # move 'intercept' to 1st position
        cols = list(df_x.columns)
        cols = [cols[-1]] + cols[:-1]
        df_x = df_x[cols]

        # convert pandas dataframe to numpy array
        x = df_x.to_numpy()
        y = df_y.to_numpy()
        return x, y, cols

    def predict(self, K, alpha, y):
        prediction = np.sign(np.dot(np.multiply(alpha, y), K))
        return prediction

    def accuracy(self, K, alpha, y, y_train):
        prediction = self.predict(K, alpha, y_train)
        accuracy = (y == prediction).mean()
        return accuracy

    def kernel_func(self, x1, x2, p):
        return np.power(np.dot(x1, x2.T), p)

    def kernelized_perceptron(self, x, y, x_val, y_val, max_iters, p):
        N, M = np.shape(x)
        alpha = np.zeros(N)
        best_so_far = (-1, 0)
        best_so_far_train = (-1, 0)

        train_accuracy = []
        val_accuracy = []

        # Compute the Gram Matrix
        K_train = self.kernel_func(x, x, p)
        K_val = self.kernel_func(x, x_val, p)
        #print(np.shape(K_val))

        for iter in range(max_iters):
            for i in range(N):
                #u = np.sum(np.multiply(alpha, np.multiply(K_train[i], y)))
                u = np.dot(K_train[i], np.multiply(alpha, y))
                if u * y[i] <= 0:
                    alpha[i] += 1

            training_acc = self.accuracy(K_train, alpha, y, y)
            val_acc = self.accuracy(K_val, alpha, y_val, y)

            if training_acc > best_so_far_train[1]:
                best_so_far_train = (iter, training_acc)

            if val_acc > best_so_far[1]:
                best_so_far = (iter, val_acc)

            print(f"Kernelized Perceptron ({iter + 1}): {training_acc}, {val_acc}")
            train_accuracy.append(training_acc)
            val_accuracy.append(val_acc)

        return train_accuracy, val_accuracy, best_so_far, best_so_far_train



obj = Perceptron()
x, y, col_names = obj.read_data("pa3_train_X.csv", "pa3_train_y.csv")
x_val, y_val, col_names = obj.read_data("pa3_dev_X.csv", "pa3_dev_y.csv")
y = y.flatten()
y_val = y_val.flatten()

max_iters = 100
p = [1,2,3,4,5]

train_acc = []
val_acc = []
best_acc = []
best_acc_train = []
for i in range(len(p)):
    train_accuracy, val_accuracy, best, best_train = obj.kernelized_perceptron(x, y, x_val, y_val, max_iters, p[i])
    train_acc.append(train_accuracy)
    val_acc.append(val_accuracy)
    best_acc.append(best)
    best_acc_train.append(best_train)
    '''
    plt.plot(range(1,101), train_accuracy, label="Training Accuracy")
    plt.plot(range(1,101), val_accuracy, linestyle='--', label="Validation Accuracy")
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    title = 'Online Kernelized Perceptron (p = {})'.format(p[i])
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()
    '''

'''
plt.plot(p, list(zip(*best_acc_train))[1], marker='^', label="Best Training Accuracy")
plt.plot(p, list(zip(*best_acc))[1], marker='o', label="Best Validation Accuracy")
plt.xticks(range(1,6))
plt.xlabel('p')
plt.ylabel('Accuracy')
plt.title('Best Training and Validation Accuracy')
plt.grid()
plt.legend()
plt.show()
'''

with open('batch_training_accuracy.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerows(train_acc)

with open('batch_validation_accuracy.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerows(val_acc)

mean_training_acc = []
mean_val_acc = []

for i in range(len(train_acc)):
    mean_training_acc.append(np.asarray(train_acc[i]).mean())
    mean_val_acc.append(np.asarray(val_acc[i]).mean())

print("Best Training accuracies:", best_acc_train)
print("Best Validation accuracies:", best_acc)
print("Mean training accuracies:", mean_training_acc)
print("Mean validation accuracies:", mean_val_acc)
