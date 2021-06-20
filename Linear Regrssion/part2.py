import pandas as pd
import numpy as np
from collections import Counter
from tabulate import tabulate
import matplotlib.pyplot as plt
import sys
import csv

class LinearRegression(object):
    def read_data(self, file_name):
        # read a csv file
        df = pd.read_csv(file_name, parse_dates=True)

        # remove the ID feature
        del df['id']

        # split the date feature to month, day, year
        df['date'] = pd.to_datetime(df['date'])
        df.insert(1, 'month', df['date'].dt.month)
        df.insert(2, 'day', df['date'].dt.day)
        df.insert(3, 'year', df['date'].dt.year)
        del df['date']
        #print(df.head(5))
        col_names = list(df.columns)

        # convert pandas dataframe to numpy array
        data = df.to_numpy()
        #print(data)
        return data[:, :-1], data[:, -1], col_names

    def gradient_descent(self, x, y, w, lam, m, n, max_iters, epsilon):
        loss = []
        norm = 0

        for i in range(max_iters):
            # y_hat: (10000 x 1)
            y_hat = np.matmul(x,w)

            gradient = 2 * np.matmul(np.transpose(x), y_hat - y) / n
            norm = np.linalg.norm(gradient)
            w = w - lam * gradient

            # calculate mean-squared error
            mse = np.sum((y_hat - y) ** 2) / n
            #if mse < 10000:
            loss.append(mse)

            if mse >= sys.float_info.max / 10:
                print(f"Diverge: {i+1}")
                #print(loss)
                return w, loss

            if i % 1000 == 0:
                print(f"iteration({i+1}) norm: {norm}")
                print(f"iteration({i+1}) mse: {mse}")

            if norm <= epsilon:
                print("number of iterations:", i+1)
                print("norm:", norm)
                print("mse:",  mse)
                return w, loss

        print("number of iterations:", i+1)
        return w, loss


def plot(axrow, x, y):
    axrow[0].plot(x, color='red')
    axrow[1].plot(y, color='green')

obj = LinearRegression()
learning_rate = [100, 10, 1e-1, 1e-2, 1e-3, 1e-4, 1e-11, 1e-12]
epsilon = 0.5
max_iters = 10000

# Calculate training MSE
training_data, training_target, _ = obj.read_data('PA1_train.csv')
N, M = np.shape(training_data)
w_mat = []
mse = []

nrows = 2
fig, axes = plt.subplots(nrows, 3, figsize=(15, 6))
fig.subplots_adjust(hspace=0.6)

for i, ax in zip(range(0,6), axes.flatten()):
    w = np.random.uniform(-0.1, 0.1, M)
    w, loss = obj.gradient_descent(training_data, training_target, w, learning_rate[i], M, N, max_iters, epsilon)
    w_mat.append(w)
    mse.append(loss[-1])

    # plot MSE
#    ax.plot(loss, label='Lambda: %s' % learning_rate[i])
#    ax.grid()
#    ax.set_xlabel('Number of Iterations')
#    ax.set_ylabel('Mean-squared Error (MSE)')
#    ax.legend()

#plt.show()
print("Training MSE:", mse)

fig, axes = plt.subplots(nrows, 3, figsize=(15, 6))
fig.subplots_adjust(hspace=0.6)

# Calculate Validation  MSE
mse_dev = []
validation_data, validation_target, _ = obj.read_data('PA1_dev.csv')
for i, ax in zip(range(0,6), axes.flatten()):
    #w = np.random.uniform(-0.1, 0.1, M)
    w = np.zeros(M)
    w, loss = obj.gradient_descent(validation_data, validation_target, w, learning_rate[i], M, N, max_iters, epsilon)
    w_mat.append(w)
    mse_dev.append(loss[-1])

    # plot MSE
#    ax.plot(loss, label='Lambda: %s' % learning_rate[i])
#    ax.grid()
#    ax.set_xlabel('Number of Iterations')
#    ax.set_ylabel('Mean-squared Error (MSE)')
#    ax.legend()

#plt.show()
print("Validation MSE:", mse_dev)

