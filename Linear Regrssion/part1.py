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
        return data[:, :-1], data[:, -1], col_names[:-1]

    def report_stats(self, data, label):
        # for numerical features, report mean, std and range
        numeric_data = np.concatenate((data[:, 1:9], data[:,10:11], data[:,13:]), axis=1)
        mean = np.mean(numeric_data, axis=0)
        std = np.std(numeric_data, axis=0)
        rg = np.ptp(numeric_data, axis=0)

        col_names = ['Mean', 'Standard Deviation', 'Range']
        numeric_features = label[1:9] + label[10:11] + label[13:22]
        merged_array = np.array([numeric_features, mean, std, rg]).T
        table = tabulate(merged_array, col_names, tablefmt="fancy_grid", floatfmt=".2f")
        print(table)

        # for categorical features, report the percentage of the examples
        waterfront = self.cal_percentage(data[:, 9])
        view = self.cal_percentage(data[:, 10])
        grade = self.cal_percentage(data[:, 12])
        zipcode = self.cal_percentage(data[:, 17])


        print("[Waterfront]", end=' ')
        for element, cnt in waterfront:
            print(f"{int(element)}: {cnt/len(data)}%", end='  ')
        print('')

        print("[View]", end=' ')
        for element, cnt in view:
            print(f"{int(element)}: {cnt/len(data)}%", end='  ')
        print('')

        print("[Grade]", end=' ')
        for element, cnt in grade:
            print(f"{int(element)}: {cnt/len(data)}%", end='  ')
        print('')

        print("[Zip code]", end=' ')
        for element, cnt in zipcode:
            print(f"{int(element)}: {cnt/len(data)}%", end='  ')
        print('')

    def cal_percentage(self, data):
        cnt = Counter()
        for val in data:
            cnt[val] += 1
        return sorted(cnt.items())

    def normalize(self, data):
        for i in range(len(data[0])):
            data[:, i] = self.normalize_vector(data[:, i])
        return data

    def normalize_vector(self, vector):
        vec_range = np.ptp(vector)
        if vec_range == 0:
            return vector
        return (vector - np.min(vector)) / vec_range

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

            if mse >= sys.float_info.max:
                print(f"Diverge: {i+1}")
                #print(loss)
                return w, loss

            if i % 1000 == 0:
                print(f"iteration({i+1}) norm: {norm}")
                print(f"iteration({i+1}) mse: {mse}")

            if norm <= epsilon:
                print("number of iterations:", i+1)
                print("norm:", norm)
                #print("loss:", loss)
                return w, loss

        print("number of iterations:", i+1)
        print("norm:", norm)
        return w, loss

    def cal_mse(self, x, y, w):
        n = len(y)
        y_hat = np.matmul(x, w)
        mse = np.sum((y_hat - y) ** 2) / n

        return mse


obj = LinearRegression()
training_data, training_target, col_names = obj.read_data('PA1_train.csv')
#obj.report_stats(training_data, col_names)
normalized_x = obj.normalize(training_data)
learning_rate = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
epsilon = 0.5
max_iters = 700000


# N: num of training examples, M: num of features
N, M = np.shape(normalized_x)
w_mat = []
mse = []
for i in range(len(learning_rate)):
    w = np.zeros(M)
    #w = np.random.uniform(-0.1, 0.1, M)
    w, loss = obj.gradient_descent(normalized_x, training_target, w, learning_rate[i], M, N, max_iters, epsilon)
    w_mat.append(w)
    mse.append(loss[-1])
'''
    # plot MSE
    plt.plot(loss, label='Lambda: %s' % learning_rate[i])
    plt.xlabel('Number of Iterations')
    plt.ylabel('Mean-squared Error (MSE)')
    plt.legend()
plt.ylim(0,100)
plt.grid()
plt.show()
'''
print(mse)


with open('w_mat.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerow(col_names)
    write.writerows(w_mat)



validation_data, validation_target, _ = obj.read_data('PA1_dev.csv')
normalized_dev = obj.normalize(validation_data)
mse_dev = []
#for i in range(1,len(learning_rate)):
for i in range(1, len(w_mat)):
    mse = obj.cal_mse(normalized_dev, validation_target, w_mat[i])
    mse_dev.append(mse)
print(mse_dev)





