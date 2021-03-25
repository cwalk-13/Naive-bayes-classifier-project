import numpy as np
import scipy.stats as stats 

from mysklearn.myclassifiers import MySimpleLinearRegressor, MyKNeighborsClassifier, MyNaiveBayesClassifier

import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable 
import mysklearn.myutils as myutils

# note: order is actual/received student value, expected/solution
# def test_simple_linear_regressor_fit():
#     #x, y dataset 1
#     np.random.seed(0)
#     x1 = list(range(0, 100))
#     y1 = [value * 2 + np.random.normal(0, 25) for value in x1]
#     sp_m, sp_b, sp_r, sp_r_p_val, sp_std_err = stats.linregress(x1, y1)
#     reg = MySimpleLinearRegressor()
#     x = []
#     for i in range(len(x1)):
#         sample = [x1[i]]
#         x.append(sample)
#     reg.fit(x, y1)

#     #x,y dataset2
#     y2 = [value ** 2 + np.random.normal(0, 25) for value in x1]
#     sp_m2, sp_b2, sp_r2, sp_r_p_val2, sp_std_err2 = stats.linregress(x1, y2)
#     reg2 = MySimpleLinearRegressor()
#     reg2.fit(x, y2)
#     assert np.isclose(reg.slope, sp_m), np.isclose(reg.intercept, sp_b) 
#     #test dataset2
#     assert np.isclose(reg2.slope, sp_m2), np.isclose(reg2.intercept, sp_b2)

# def test_simple_linear_regressor_predict():
#     np.random.seed(0)
#     x1 = list(range(0, 100))
#     x = []
#     for i in range(len(x1)):
#         sample = [x1[i]]
#         x.append(sample)
    
#     y1 = [value * 2 for value in x1]
#     reg = MySimpleLinearRegressor()
#     reg.fit(x, y1)
#     x1_test = x[0:100:3]
#     y1_vals = y1[0:100:3]
#     y1_predict = reg.predict(x1_test)
#     assert np.allclose(y1_vals, y1_predict)

#     y2 = [value * 0.5 + 5 for value in x1]
#     reg2 = MySimpleLinearRegressor()
#     reg2.fit(x, y2)
#     x1_test = x[0:100:3]
#     y2_vals = y2[0:100:3]
#     y2_predict = reg2.predict(x1_test)
#     assert np.allclose(y2_vals, y2_predict)

# def test_kneighbors_classifier_kneighbors():
#     train0 = [
#         [7, 7],
#         [7, 4],
#         [3, 4],
#         [1, 4]
#     ]
#     train0_labels = ["bad", "bad", "good", "good"]
#     test0 = [[3, 7]]
#     knn0 = MyKNeighborsClassifier()
#     knn0.fit(train0, train0_labels)
#     dists0, indices0 = knn0.kneighbors(test0)
#     real_indices = [[2, 3, 0]]
#     assert indices0 == real_indices
#     train1 = [
#         [3, 2],
#         [6, 6],
#         [4, 1],
#         [4, 4],
#         [1, 2],
#         [2, 0],
#         [0, 3],
#         [1, 6]
#     ]
#     train1_labels = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
#     test1 = [[2, 3]]
#     knn1 = MyKNeighborsClassifier()
#     knn1.fit(train1, train1_labels)
#     dists1, indices1 = knn1.kneighbors(test1)
#     real_indices = [[0, 4, 6]]

#     assert indices1 == real_indices # TODO: fix this

# def test_kneighbors_classifier_predict():
#     train0 = [
#         [7, 7],
#         [7, 4],
#         [3, 4],
#         [1, 4]
#     ]
#     train0_labels = ["bad", "bad", "good", "good"]
#     test0 = [[3, 7]]
#     knn0 = MyKNeighborsClassifier()
#     knn0.fit(train0, train0_labels)
#     predicted0 = knn0.predict(test0)
#     actual = ["good"] 
#     assert predicted0 == actual
   
#     train = [
#         [3, 2],
#         [6, 6],
#         [4, 1],
#         [4, 4],
#         [1, 2],
#         [2, 0],
#         [0, 3],
#         [1, 6]
#     ]
#     train_labels = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
#     test = [[2, 3]]
#     knn1 = MyKNeighborsClassifier()
#     knn1.fit(train, train_labels)
#     predicted = knn1.predict(test)
#     actual = ["yes"] 
#     assert predicted == actual # TODO: fix this

def test_naive_bayes_classifier_fit():
    train = [
        [1,	5],	
        [2,	6],	
        [1,	5],	
        [1,	5],
        [1,	6],
        [2,	6],
        [1,	5],
        [1,	6]
    ]
    y = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    nb = MyNaiveBayesClassifier()
    nb.fit(train, y)
    assert nb.priors == [["yes", 5/8],["no", 3/8]]
    assert nb.posteriors == [[0, ['yes', ['1', 0.8], ['2', 0.2]], ['no', ['1', 2/3], ['2', 1/3]]], \
                            [1, ['yes', ['5', 0.4], ['6', 0.6]], ['no', ['5', 2/3], ['6', 1/3]]]]

    # RQ5 (fake) iPhone purchases dataset
    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]
    mypy = MyPyTable(iphone_col_names, iphone_table)
    y2 = myutils.get_mypycol(mypy, "buys_iphone")
    nb2 = MyNaiveBayesClassifier()
    nb2.fit(iphone_table, y2)
    assert nb2.priors == [["no", 1/3], ["yes", 2/3]]
    nb2_posts = [
        [0, ['no', ['1', 3/15], ['2', 2/15]], ['yes', ['1', 2/15], ['2', 8/15]]], 
        [1, ['no', ['3', 2/15], ['2', 2/15], ['1', 2/3]], ['yes', ['3', 3/15], ['2', 4/15], ['1', 3/15]]], 
        [2, ['no', ['fair', 2/15], ['excellent', 3/15]], ['yes', ['fair', 7/15], ['excellent', 3/15]]], 
        [3, ['no', ['no', 1/3], ['yes', 0.0]], ['yes', ['no', 0.0], ['yes', 2/3]]]
    ]
    # assert nb2.posteriors == nb2_posts
    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"], 
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]
    mypy2 = MyPyTable(train_col_names, train_table)
    y3 = myutils.get_mypycol(mypy2, "class")
    nb3 = MyNaiveBayesClassifier()
    nb3.fit(iphone_table, y3)


def test_naive_bayes_classifier_predict():
    train = [
        [1,	5],	
        [2,	6],	
        [1,	5],	
        [1,	5],
        [1,	6],
        [2,	6],
        [1,	5],
        [1,	6]
    ]
    y = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    nb = MyNaiveBayesClassifier()
    nb.fit(train, y)

    pred = nb.predict([[1,5]])
    
    assert pred == ["yes"] # TODO: fix this
    # RQ5 (fake) iPhone purchases dataset
    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]
    mypy = MyPyTable(iphone_col_names, iphone_table)
    y2 = myutils.get_mypycol(mypy, "buys_iphone")
    nb2 = MyNaiveBayesClassifier()
    nb2.fit(iphone_table, y2)
    pred2 = nb2.predict([[1, 2, "fair"]])

    assert pred2 == ["yes"]
    
    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"], 
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]
    mypy2 = MyPyTable(train_col_names, train_table)
    y3 = myutils.get_mypycol(mypy2, "class")
    nb3 = MyNaiveBayesClassifier()
    nb3.fit(train_table, y3)
    nb3.fit(train_table, y3)
    pred3 = nb3.predict([["weekday", "winter", "high", "heavy"]])

    assert pred3 == ["cancelled"]