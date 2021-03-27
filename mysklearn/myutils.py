# TODO: your reusable general-purpose functions here
import math 
import numpy as np
import importlib
import copy
import random
import mysklearn.mypytable
importlib.reload(mysklearn.mypytable)
from mysklearn.mypytable import MyPyTable
from tabulate import tabulate

# import mysklearn.myutils
# importlib.reload(mysklearn.myutils)
# import mysklearn.myutils as myutils

# # uncomment once you paste your mypytable.py into mysklearn package


# import mysklearn.myclassifiers
# importlib.reload(mysklearn.myclassifiers)
# from mysklearn.myclassifiers import MyKNeighborsClassifier, MySimpleLinearRegressor

# import mysklearn.myevaluation
# importlib.reload(mysklearn.myevaluation)
# import mysklearn.myevaluation as myevaluation


def get_column(table, header, col_name):
    col_index = header.index(col_name)
    col = []
    for row in table: 
        # ignore missing values ("NA")
        if row[col_index] != "NA":
            col.append(row[col_index])
    return col

def get_col_byindex(table, i):
    col = []
    for row in table: 
        # ignore missing values ("NA")
        if row[i] != "NA":
            col.append(row[i])
    return col

def get_frequencies(table, header, col_name):
    col = get_column(table, header, col_name)

    col.sort() # inplace
    values = []
    counts = []

    for value in col:
        if value not in values:
            # first time we have seen this value
            values.append(value)
            counts.append(1)
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    return values, counts

def group_by(table, header, group_by_col_name):
    col = get_column(table, header, group_by_col_name)
    col_index = header.index(group_by_col_name)
    
    # we need the unique values for our group by column
    group_names = sorted(list(set(col))) # e.g. 74, 75, 76, 77
    group_subtables = [[] for _ in group_names] # [[], [], [], []]
    
    # algorithm: walk through each row and assign it to the appropriate
    # subtable based on its group_by_col_name value
    for row in table:
        group_by_value = row[col_index]
        # which subtable to put this row in?
        group_index = group_names.index(group_by_value)
        group_subtables[group_index].append(row.copy()) # shallow copy
    
    return group_names, group_subtables

def compute_equal_width_cutoffs(values, num_bins):
    # first compute the range of the values
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins 
    # bin_width is likely a float
    # if your application allows for ints, use them
    # we will use floats
    # np.arange() is like the built in range() but for floats
    cutoffs = list(np.arange(min(values), max(values), bin_width)) 
    cutoffs.append(max(values))
    # optionally: might want to round
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs 
    
def compute_bin_frequencies(values, cutoffs):
    freqs = [0 for _ in range(len(cutoffs) - 1)]

    for val in values:
        if val == max(values):
            freqs[-1] += 1
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= val < cutoffs[i + 1]:
                    freqs[i] += 1

    return freqs

def compute_slope_intercept(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    # y = mx + b => b = y - mx
    b = mean_y - m * mean_x
    return m, b 

def compute_euclidean_distance(v1, v2):
    print(v1, v2)
    assert len(v1) == len(v2)

    dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return dist 

#pa3 add-ons
#
#

def conv_num(mypy):
    mypy.convert_to_numeric
    pass

def load_data(filename):
    mypytable = MyPyTable()
    mypytable.load_from_file(filename)
    return mypytable

def get_min_max(values):
    return min(values), max(values)

def binary_freq(mypy, col_name):
    mypy.convert_to_numeric()
    col = get_mypycol(mypy, col_name)
    freq = 0
    for i in range(len(col)):
        if col[i] == 1:
            freq += 1

    return col_name, freq

def percent_compare(mypy, col_names, total, get_sum=True):
    conv_num(mypy)
    percentages = []
    if get_sum == False:
        for i in range(len(col_names)):
            col = get_mypycol(mypy, col_names[i])
            col2 = []
            for j in range(len(col)):
                if col[j] != 0:
                    col2.append(col[j])
            col_total = len(col2)
            prcnt = col_total / total
            percentages.append(prcnt)
    if get_sum == True:
        for i in range(len(col_names)):
            col = get_mypycol(mypy, col_names[i])
            col_total = sum(col)
            prcnt = col_total / total
            percentages.append(prcnt)
    return col_names, percentages

# pa4 add-ons
#
#
def mpg_rating(val):
    rating = 0
    if val <=13:
        rating = 1
    elif val == 14:
        rating = 2
    elif 15 <= val < 17:
        rating = 3
    elif 17 <= val < 20:
        rating = 4
    elif 20 <= val < 24:
        rating = 5
    elif 24 <= val < 27:
        rating = 6
    elif 27 <= val < 31:
        rating = 7
    elif 31 <= val < 37:
        rating = 8
    elif 37 <= val < 45:
        rating = 9
    elif val >= 45:
        rating = 10
    return rating

def get_freq_str(col):
    
    header = ["y"]
    col_mypy = MyPyTable(header, col)

    dups = col_mypy.ordered_col(header)
    values = []
    counts = []

    for value in dups:
        if value not in values:
            # first time we have seen this value
            values.append(str(value))
            counts.append(1)
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    return values, counts

def get_accuracy(actual, pred):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == pred[i]:
            correct+=1
    return correct/len(actual)

def get_mypycol(mypy, col_name):
    return mypy.get_column(col_name, False)

def get_freq_1col(col):
    col.sort() # inplace
    values = []
    counts = []

    for value in col:
        if value not in values:
            # first time we have seen this value
            values.append(value)
            counts.append(1)
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    return values, counts

def get_rand_rows(table, num_rows):
    rand_rows = []
    for i in range(num_rows):
        rand_rows.append(table.data[random.randint(0,len(table.data))-1])
    return rand_rows

def rating(mpg):
    if mpg < 14:
        return 1
    elif mpg < 15:
        return 2
    elif mpg < 17:
        return 3
    elif mpg < 20:
        return 4
    elif mpg < 24:
        return 5
    elif mpg < 27:
        return 6
    elif mpg < 31:
        return 7
    elif mpg < 37:
        return 8
    elif mpg < 45:
        return 9
    return 10


def categorize_weight(val):
    if val < 2000:
        weight = 1
    elif val < 2500:
        weight = 2
    elif val < 3000:
        weight = 3
    elif val < 3500:
        weight = 4
    else:
        weight = 5
    return weight

def convert_weights(weight):
        res = []
        for val in weight:
            res.append(categorize_weight(val))
        return res

def print_results(rows, actual, predicted):
        for i in range(len(rows)):
            print('instance:', rows[i])
            print('class:', predicted[i], 'actual:', actual[i])
            
def mpg_to_rating(mpg):
    for i in range(len(mpg)):
        mpg[i] = rating(mpg[i])
    return mpg

def folds_to_train(x, y, train_folds, test_folds):
    X_train = []
    y_train = []
    for row in train_folds:
        for i in row:
            X_train.append(x[i])
            y_train.append(y[i])

    X_test = []
    y_test = []
    for row in test_folds:
        for i in row:
            X_test.append(x[i])
            y_test.append(y[i])

    return X_train, y_train, X_test, y_test

def add_config_stats(matrix):
    del matrix[0]
    for i,row in enumerate(matrix):
        row[0] = i+1
        row.append(sum(row))
        row.append(round(row[i+1]/row[-1]*100,2))
        
def titanic_matrix(matrix):
    for i,row in enumerate(matrix):
        row.append(sum(row))
        row.append(round(row[i]/row[-1]*100,2))
        row.insert(0, i+1)
    matrix.append(['Total', matrix[0][1]+matrix[1][1], matrix[0][2]+matrix[1][2], matrix[0][3]+matrix[1][3], \
                   round(((matrix[0][1]+matrix[1][2])/(matrix[0][3]+matrix[1][3])*100),2)])

def print_tabulate(table, headers):
    print(tabulate(table, headers, tablefmt="rst"))