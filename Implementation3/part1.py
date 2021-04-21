import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import random

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

    def predict(self, x, w):
        prediction = np.sign(np.dot(x, w))
        return prediction

    def accuracy(self, x, y, w):
        prediction = self.predict(x, w)
        accuracy = (y == prediction).mean()
        return accuracy

    def perceptron(self, x, y, x_val, y_val, max_iters):
        N, M = np.shape(x)
        w_online = w_average = np.zeros(M)
        best_so_far_online = (-1, 0)
        best_so_far_avg = (-1, 0)

        online_train_accuracy = []
        online_val_accuracy = []
        average_train_accuracy = []
        average_val_accuracy = []

        s = 1
        for iter in range(max_iters):
            #indices = np.random.permutation(len(x))
            #x = x[indices]
            #y = y[indices]
            for i in range(N):
                score = y[i]*np.dot(w_online.T, x[i])
                if score <= 0:
                    w_online += y[i] * x[i]
                w_average = (s*w_average + w_online) / (s+1)
                s += 1

            online_training_acc = self.accuracy(x, y, w_online)
            online_val_acc = self.accuracy(x_val, y_val, w_online)
            avg_training_acc = self.accuracy(x, y, w_average)
            avg_val_acc = self.accuracy(x_val, y_val, w_average)
            if online_val_acc > best_so_far_online[1]:
                best_so_far_online = (iter, online_val_acc)

            if avg_val_acc > best_so_far_avg[1]:
                best_so_far_avg = (iter, avg_val_acc)

            print(f"Online Perceptron  ({iter+1}): {online_training_acc}, {online_val_acc}")
            print(f"Average Perceptron ({iter+1}): {avg_training_acc}, {avg_val_acc}")

            online_train_accuracy.append(online_training_acc)
            online_val_accuracy.append(online_val_acc)
            average_train_accuracy.append(avg_training_acc)
            average_val_accuracy.append(avg_val_acc)

        return online_train_accuracy, online_val_accuracy, average_train_accuracy, average_val_accuracy, best_so_far_online, best_so_far_avg


obj = Perceptron()
x, y, col_names = obj.read_data("pa3_train_X.csv", "pa3_train_y.csv")
x_val, y_val, col_names = obj.read_data("pa3_dev_X.csv", "pa3_dev_y.csv")
y = y.flatten()
y_val = y_val.flatten()

max_iters = 100
online_train_accuracy, online_val_accuracy, average_train_accuracy, average_val_accuracy, best_online, best_avg = obj.perceptron(x, y, x_val, y_val, max_iters)
'''
plt.plot(range(1,101), online_train_accuracy, label="Training Accuracy")
plt.plot(range(1,101), online_val_accuracy, linestyle='--', label="Validation Accuracy")
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy')
plt.title('Online Perceptron')
plt.grid()
plt.legend()
plt.show()

plt.plot(range(1,101), average_train_accuracy, label="Training Accuracy")
plt.plot(range(1,101), average_val_accuracy, linestyle='--', label="Validation Accuracy")
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy')
plt.title('Average Perceptron')
plt.grid()
plt.legend()
plt.show()
'''
print(f"Highest validation accuracy (online): {best_online[1]} at iteration {best_online[0]+1}")
print(f"Highest validation accuracy (average): {best_avg[1]} at iteration {best_avg[0]+1}")