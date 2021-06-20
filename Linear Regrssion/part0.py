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
        condition = self.cal_percentage(data[:, 11])
        grade = self.cal_percentage(data[:, 12])
        zipcode = self.cal_percentage(data[:, 17])


        print("[Waterfront]", end=' ')
        for element, cnt in waterfront:
            print(f"{int(element)}: {cnt/len(data)}%", end='  ')
        print('')

        print("[Condition]", end=' ')
        for element, cnt in condition:
            print(f"{int(element)}: {cnt/len(data)}%", end='  ')
        print('')

        print("[Grade]", end=' ')
        for element, cnt in grade:
            print(f"{int(element)}: {cnt/len(data)}%", end='  ')
        print('')

        i = 0
        print("[Zipcode]")
        for element, cnt in zipcode:
            print(f"{int(element)}: {cnt / len(data):.4f}%", end='  ')
            if i % 7 == 6:
                print('')
            i += 1
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


obj = LinearRegression()
training_data, training_target, col_names = obj.read_data('PA1_train.csv')
obj.report_stats(training_data, col_names)
normalized_x = obj.normalize(training_data)



