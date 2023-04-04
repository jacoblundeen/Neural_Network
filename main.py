"""
605.649 Introduction to Machine Learning
Dr. Sheppard
Programming Project #03
20210719
Jacob M. Lundeen

The purpose of this assignment is to give you a chance to get some hands-on experience learning decision
trees for classification and regression. This time around, we are not going to use anything from the module
on rule induction; however, you might want to examine the rules learned for your trees to see if they “make
sense.” Specifically, you will be implementing a standard univariate (i.e., axis-parallel) decision tree and will
compare the performance of the trees when grown to completion on trees that use either early stopping (for
regression trees) or reduced error pruning (for classification trees).

Let’s talk about the numeric attributes. There are two ways of handling them. The first involves
discretizing (binning), similar to what you were doing in earlier assignments. This is not the preferred
approach, so we ask that you avoid binning these attributes. Instead, the second and preferred approach
is to sort the data on the attribute and consider possible binary splits at midpoints between adjacent data
points. Note that this could lead to a lot of possible splits. One way to reduce that is to consider midpoints
between data where the class changes. For regression, there is no corresponding method, so you should
consider splits near the middle of the sorted range and not consider all possible.

For decision trees, it should not matter whether you have categorical or numeric attributes, but you need
to remember to keep track of which is which. In addition, you need to implement that gain-ratio criterion
for splitting in your classification trees. Go ahead and eliminate features that act as unique identifiers of the
data points.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math
import time
from numpy import log2 as log
import pprint
import warnings
import random
from statistics import mean
from Log_Reg import Log_Reg
from Lin_Reg import LinearRegression
from FFNN import Neural_Network
from autoNN import Auto_NN

warnings.filterwarnings('ignore')

eps = np.finfo(float).eps


# Function to read in the data set. For those data sets that do not have header rows, this will accept a list of
# column names.
def read_data(data, names=[], fillna=True):
    if not names:
        return pd.read_csv(data)
    if not fillna:
        return pd.read_csv(data, names=names)
    else:
        return pd.read_csv(data, names=names, na_values='?')


# Function to handle missing data. Only data set that needed handling was the cancer data set.
def missing_values(data, column_name):
    data[column_name].fillna(value=data[column_name].mean(), inplace=True)


# Function to change categorical data into numerical. If the data is not ordinal, then dummy variables are used. If the
# data is ordinal, then one hot encoding is used. These had to be hard coded for the cars and forest data sets.
def cat_data(data, var_name='', ordinal=False, data_name=''):
    if ordinal:
        if data_name == 'cars':
            buy_main_mapper = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
            door_mapper = {'2': 2, '3': 3, '4': 4, '5more': 5}
            per_mapper = {'2': 2, '4': 4, 'more': 5}
            lug_mapper = {'small': 0, 'med': 1, 'big': 2}
            saf_mapper = {'low': 0, 'med': 1, 'high': 2}
            class_mapper = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
            mapper = [buy_main_mapper, buy_main_mapper, door_mapper, per_mapper, lug_mapper, saf_mapper, class_mapper]
            count = 0
            for col in data.columns:
                data[col] = data[col].replace(mapper[count])
                count += 1
            return data
        elif data_name == 'forest':
            month_mapper = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
                            'oct': 10, 'nov': 11, 'dec': 12}
            day_mapper = {'sun': 1, 'mon': 2, 'tue': 3, 'wed': 4, 'thu': 5, 'fri': 6, 'sat': 7}
            data.month = data.month.replace(month_mapper)
            data.day = data.day.replace(day_mapper)
            return data
        elif data_name == 'cancer':
            class_mapper = {2: 0, 4: 1}
            data[var_name] = data[var_name].replace(class_mapper)
            return data
    else:
        return pd.get_dummies(data, columns=var_name, prefix=var_name)


# The discretization function. Provides the user the ability to do either equal width or equal frequency by passing
# 'False' to the equal_width variable. The bin size defaults to 20, and the function will discretize with a specific
# variable or the entire data set.
def discret(data, equal_width=True, num_bin=20, feature=""):
    if equal_width:
        if not feature:
            for col in data.columns:
                data[col] = pd.cut(x=data[col], bins=num_bin)
            return data
        else:
            data[feature] = pd.cut(x=data[feature], bins=num_bin)
            return data
    else:
        if not feature:
            for col in data.columns:
                data[col] = pd.qcut(x=data[col], q=num_bin, duplicates='drop')
            return data
        else:
            data[feature] = pd.qcut(x=data[feature], q=num_bin)
            return data


# The standardization() function performs z-score standardization on a given train and test set. The function
# will standardize either an individual feature or the entire data set. If the standard deviation of a variable is 0,
# then the variable is constant and adds no information to the regression, so it can be dropped from the data set.
def standardization(train, test=pd.DataFrame(), feature=''):
    if test.empty:
        for col in train.columns:
            if train[col].dtype == "object":
                continue
            else:
                if train[col].std() == 0:
                    train.drop(col, axis=1, inplace=True)
                else:
                    train[col] = (train[col] - train[col].mean()) / train[col].std()
        return train
    elif not feature:
        for col in train.columns:
            if train[col].std() == 0:
                train.drop(col, axis=1, inplace=True)
                test.drop(col, axis=1, inplace=True)
            else:
                test[col] = (test[col] - train[col].mean()) / train[col].std()
                train[col] = (train[col] - train[col].mean()) / train[col].std()
        return train, test
    else:
        test[feature] = (test[feature] - train[feature].mean()) / train[feature].std()
        train[feature] = (train[feature] - train[feature].mean()) / train[feature].std()
        return train, test


# The class_split() function handles splitting and stratifying classification data. Returns validation set and
# data_splits.
def class_split(data, k, class_var, validation=False):
    # Group the data set by class variable using pd.groupby()
    grouped = data.groupby([class_var])
    grouped_l = []
    data_splits = []
    # Create stratified validation set using np.split(). 20% from each group is appended to the validation set, the rest
    # will be used for the k-folds.
    if validation:
        grouped_val = []
        grouped_dat = []
        for name, group in grouped:
            val, dat = np.split(group, [int(0.2 * len(group))])
            grouped_val.append(val)
            grouped_dat.append(dat)
        # Split the groups into k folds
        for i in range(len(grouped_dat)):
            grouped_l.append(np.array_split(grouped_dat[i], k))
    else:
        for name, group in grouped:
            grouped_l.append(np.array_split(group.iloc[np.random.permutation(np.arange(len(group)))], k))
        for i in range(len(grouped_l)):
            for j in range(len(grouped_l[i])):
                grouped_l[i][j].reset_index(inplace=True, drop=True)
        for i in range(k):
            temp = grouped_l[0][i]
            for j in range(1, len(grouped_l)):
                temp = pd.concat([temp, grouped_l[j][i]], ignore_index=True)
            data_splits.append(temp)
    # Reset indices of the folds
    for item in range(len(grouped_l)):
        for jitem in range(len(grouped_l[item])):
            grouped_l[item][jitem].reset_index(inplace=True, drop=True)
    # Combine folds from each group to create stratified folds
    for item in range(k):
        tempo = grouped_l[0][item]
        for jitem in range(1, len(grouped_l)):
            tempo = pd.concat([tempo, grouped_l[jitem][item]], ignore_index=True)
        data_splits.append(tempo)
    if validation:
        grouped_val = pd.concat(grouped_val)
        return grouped_val, pd.concat(data_splits)
    else:
        return data_splits[0], data_splits[1]



# The reg_split() function creates the k-folds for regression data.
def reg_split(data, k, validation=False):
    # Randomize the data first
    df = data.sample(frac=1, random_state=42).reset_index(drop=True)
    # If a validation set is required, divide the data set 20/80 and return the sets
    if validation:
        val_fold, data_fold = np.split(df, [int(.2 * len(df))])
        if k == 1:
            return val_fold, data_fold.reset_index(drop=True)
        else:
            data_fold = np.array_split(data_fold, k)
            return val_fold, data_fold
    # If no validation set is required, split the data by the requested k
    else:
        data_fold = np.array_split(df, k)
        val_fold = 0
        return data_fold[0], data_fold[1]


# The k2_cross() function performs Kx2 cross validation.
def k2_cross(data, k, class_var, pred_type, max_iter, alpha, num_nodes):
    results = []
    count = 0
    if pred_type == 'regression':
        data = standardization(data)
    # num_nodes = [10, 5]
    # As we loop over k, we randomize each loop and then split the data 50/50 into train and test sets (standardizing
    # when doing regression). The learning algorithm is trained on the training set first and then tested on the test
    # set. They are then flipped (trained on the test set and tested on the train set). So we get 2k experiments.
    while count < 5:
        rand_df = data.sample(frac=1, random_state=42).reset_index(drop=True)
        dfs = np.array_split(rand_df, 2)
        train = dfs[0]
        test = dfs[1]
        if pred_type == 'classification':
            true = test[class_var]
            model = Neural_Network(train.copy(), class_var, pred_type, num_nodes, alpha=alpha)
            # model = Auto_NN(train.copy(), class_var, pred_type, num_nodes, alpha=alpha)
            model.gradient_descent(max_iter)
            pred = model.make_predictions(test.drop(class_var, axis=1).values.T)
            results.append(model.get_accuracy(pred, true))

            true = train[class_var]
            model = Neural_Network(test.copy(), class_var, pred_type, num_nodes, alpha=alpha)
            # model = Auto_NN(test.copy(), class_var, pred_type, num_nodes, alpha=alpha)
            model.gradient_descent(max_iter)
            pred = model.make_predictions(train.drop(class_var, axis=1).values.T)
            results.append(model.get_accuracy(pred, true))
        else:
            true = test[class_var]
            model = Auto_NN(train.copy(), class_var, pred_type, num_nodes, alpha=alpha)
            # model = Neural_Network(train.copy(), class_var, pred_type, num_nodes, alpha=alpha)
            model.gradient_descent(max_iter)
            pred = model.make_predictions(test.drop(class_var, axis=1).values.T)
            results.append(eval_metrics(true, pred, pred_type))

            true = train[class_var]
            model = Auto_NN(train.copy(), class_var, pred_type, num_nodes, alpha=alpha)
            # model = Neural_Network(test.copy(), class_var, pred_type, num_nodes, alpha=alpha)
            model.gradient_descent(max_iter)
            pred = model.make_predictions(train.drop(class_var, axis=1).values.T)
            results.append(eval_metrics(true, pred, pred_type))
        count += 1
    if pred_type == 'classification':
        return mean(results)
    else:
        metric1 = []
        metric2 = []
        metric3 = []
        for lst in results:
            metric1.append(lst[0])
            metric2.append(lst[1])
            metric3.append(lst[2])
        final_metrics = [mean(metric1), mean(metric2), mean(metric3)]
        return final_metrics


# Function to create the validation set if desired. It will randomly sample the data set (defaulted to 20%), remove
# those values from the original data set, reset their indices, and return them both.
def validation_set(data, size=0.2):
    val = data.sample(frac=size)
    data_splits = data.drop(val.index)
    val.reset_index(inplace=True, drop=True)
    data_splits.reset_index(inplace=True, drop=True)
    return data_splits, val


# Calculate the R2 score
def r2_score(true, pred):
    if len(pred) == 0:
        return 0
    else:
        mean_true = np.mean(true)
        ss_r = true.subtract(pred) ** 2
        ss_r = ss_r.sum()
        ss_t = true.subtract(mean_true) ** 2
        ss_t = ss_t.sum()
        if ss_t == 0:
            r2 = 1
        else:
            r2 = round(1 - (ss_r / ss_t), 3)
        return r2


# Metrics functions. This will calculate the desired metrics for either regression or classification. Classification
# had to be protected from dividing by zero.
def eval_metrics(true, predicted, eval_type='regression'):
    # For regression, we create the correlation matrix and then calculate the R2, Person's Correlation, and MSE.
    if len(predicted) == 0:
        return 0, 0, 0
    elif eval_type == 'regression':
        r2_s = r2_score(true, predicted)
        persons = round(pd.Series(true).corr(pd.Series(predicted)), 3)
        mse = round(np.square(np.subtract(true, predicted)).mean(), 3)
        return r2_s, persons, mse
    # For classification, we calculate Precision, Recall, and F1 scores.
    else:
        total_examp = len(true)
        precision = []
        recall = []
        f_1 = []
        count = 0
        for label in np.unique(true):
            class_len = np.sum(true == label)
            true_pos = np.sum((true == label) & (predicted == label))
            true_neg = np.sum((true != label) & (predicted != label))
            false_pos = np.sum((true != label) & (predicted == label))
            false_neg = np.sum((true == label) & (predicted != label))
            if true_pos & false_pos == 0:
                precision.append(0)
            else:
                precision.append(true_pos / (true_pos + false_pos))
            if true_pos + false_neg == 0:
                recall.append(0)
            else:
                recall.append(true_pos / (true_pos + false_neg))
            if precision[count] + recall[count] == 0:
                f_1.append(0)
            else:
                if len(np.unique(true)) > 1:
                    f_1.append((class_len / total_examp) * 2 * (precision[count] * recall[count]) / (
                            precision[count] + recall[count]))
                else:
                    f_1.append(2 * (precision[count] * recall[count]) / (precision[count] + recall[count]))
            count += 1
        if count > 1:
            return mean(precision), mean(recall), mean(f_1)
        else:
            return sum(precision), sum(recall), sum(f_1)


# The hyper_tune() function is used to tune the hyperparameters of the decision tree
def hyper_tune(data, class_var, pred_type):
    # Create lists of values for max depth and theta and loop, over to determine parameters (lowest MSE or highest F1)
    max_iter = [100, 500, 1000]
    # alpha = [round(i, 3) for i in np.linspace(0.001, 0.1, 10)]
    alpha = [0.001, 0.0001, 0.00001, 0.0000001]
    num_node1 = np.arange(1, data.shape[1], 2)
    num_node2 = np.arange(1, (data.shape[1] + 10), 3)
    mu = [round(i, 3) for i in np.linspace(0.01, 1, 5)]
    if pred_type == 'regression':
        df_hyper = pd.DataFrame(columns=['Max Iter', 'Alpha', 'Num Node 1', 'Num Node 2', 'MSE'])
        for mi in max_iter:
            for a in alpha:
                for num1 in num_node1:
                    for num2 in num_node2:
                        results = k2_cross(data, 1, class_var, pred_type, num_nodes=[num1, num2], max_iter=mi, alpha=a)
                        temp = pd.DataFrame(data={'Max Iter': mi, 'Alpha': a, 'Num Node 1': num1, 'Num Node 2': num2, 'MSE': round(results[2], 3)}, index=[0])
                        df_hyper = pd.concat([df_hyper, temp], ignore_index=True)
        df_hyper.to_csv('abalone_ann.csv')
        mse_min = df_hyper[df_hyper.MSE == df_hyper.MSE.min()]
        mse_min.reset_index(drop=True)
        return mse_min.iat[0, 0], mse_min.iat[0, 1], mse_min.iat[0, 2], mse_min.iat[0, 3], mse_min.iat[0, 4]
    else:
        df_hyper = pd.DataFrame(columns=['Max Iter', 'Alpha', 'Num Node 1', 'Num Node 2', 'Accuracy'])
        # mi = 1500
        # a = 0.001
        for mi in max_iter:
            for a in alpha:
                for num1 in num_node1:
                    for num2 in num_node2:
                        results = k2_cross(data, 1, class_var, pred_type, num_nodes=[num1, num2], max_iter=mi, alpha=a)
                        temp = pd.DataFrame(data={'Max Iter': mi, 'Alpha': a, 'Num Node 1': num1, 'Num Node 2': num2, 'Accuracy': round(results, 3)}, index=[0])
                        df_hyper = pd.concat([df_hyper, temp], ignore_index=True)
        df_hyper.to_csv('cars_nn.csv')
        f1_max = df_hyper[df_hyper.Accuracy == df_hyper.Accuracy.max()]
        f1_max.reset_index(drop=True)
        return f1_max.iat[0, 0], f1_max.iat[0, 1], f1_max.iat[0, 2], f1_max.iat[0, 3], f1_max.iat[0, 4]


# Function to protect against division by zero
def divide_zero(x, y):
    return x / y if y else 0


# Return subset of data based on current splitting node and unique value
def get_subset(data, node, value):
    return data[data[node] == value].reset_index(drop=True)


# Function to handle all pre-processing of the six data sets. Can have the data encoded if desired.
def pre_process(encode=False):
    # Read in data. Attribute names have to be hardcoded for all data sets minus the forest data set.
    ab_names = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight',
                'shell_weight', 'rings']
    abalone = read_data('abalone.data', ab_names)
    abalone = move_targetVar(abalone, 'rings')
    cancer_names = ['code_num', 'clump_thick', 'unif_size', 'unif_shape', 'adhesion', 'epithelial_size', 'bare_nuclei',
                    'bland_chromatin', 'norm_nucleoli', 'mitosis', 'class']
    cancer = read_data('breast-cancer-wisconsin.data', cancer_names)
    cancer.drop('code_num', axis=1, inplace=True)
    cancer = move_targetVar(cancer, 'class')
    car_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    cars = read_data('car.data', car_names)
    cars = move_targetVar(cars, 'class')
    forest = read_data('forestfires.csv')
    forest = move_targetVar(forest, 'area')
    forest['area'] = np.log2(forest['area'] + eps)
    house_names = ['class', 'infants', 'water_sharing', 'adoption_budget', 'physician_fee', 'salvador_aid',
                   'religious_schools', 'anti_sat_ban', 'aid_nic_contras', 'mx_missile', 'immigration',
                   'synfuels_cutback', 'edu_spending', 'superfund_sue', 'crime', 'duty_free', 'export_admin_africa']
    house = read_data('house-votes-84.data', house_names, fillna=False)
    house = move_targetVar(house, 'class')
    machine_names = ['vendor', 'model', 'myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax', 'prp', 'erp']
    machine = read_data('machine.data', machine_names)
    machine.drop('model', axis=1, inplace=True)
    machine = move_targetVar(machine, 'erp')

    # Handle missing values; only data set that needs it is the cancer data set.
    missing_values(cancer, 'bare_nuclei')

    if encode:
        # Categorical Data; 6 of the 6 data sets have categorical data that need to be encoded for the ML algorithms.
        abalone = cat_data(abalone, [ab_names[0]])
        abalone = move_targetVar(abalone, 'rings')
        cancer = cat_data(cancer, var_name='class', data_name='cancer', ordinal=True)
        cars = cat_data(cars, var_name=car_names, data_name='cars', ordinal=True)
        forest = cat_data(forest, data_name='forest', ordinal=True)
        house = cat_data(house, var_name=list(house_names))
        house = move_targetVar(house, 'class_republican')
        machine = cat_data(machine, var_name=[machine_names[0]])
        machine = move_targetVar(machine, 'erp')

    return abalone, cancer, cars, forest, house, machine


# Move target, or class, variable to end of dataframe.
def move_targetVar(data, target):
    cols = data.columns.tolist()
    cols.insert(len(cols), cols.pop(cols.index(target)))
    data = data.reindex(columns=cols)
    return data


# Return the MSE
def mse(true, pred):
    if len(true) == 0:
        return 0
    else:
        return round(np.square(true.subtract(pred)).mean(), 3)


# Main function.
if __name__ == '__main__':
    np.random.seed(420)
    abalone, cancer, cars, forest, house, machine = pre_process(True)
    data = machine
    dname = 'machine'
    target_variable = 'erp'
    pred_type = 'regression'
    machine = standardization(machine)

    test_set, train_set = reg_split(data, 2, False)
    model = LinearRegression(eta=0.01, iterations=400)
    model.fit(train_set, target_variable)
    y_hat = model.predict(test_set.drop(target_variable, axis=1))
    _, _, mse = eval_metrics(test_set[target_variable], y_hat)
    print('This is the MSE for Linear Regression on the Machine data set: ', mse)

    model = Neural_Network(data, target_variable, pred_type, [31, 11], 0.00001)
    model.gradient_descent(100)
    pred = model.make_predictions(test_set.drop(target_variable, axis=1).values.T)
    _, _, mse = eval_metrics(test_set[target_variable], pred)
    print('This is the MSE for the Feedforward Neural Network on the Machine data set: ', mse)

    model = Auto_NN(train_set, target_variable, pred_type, [1, 26], 0.000001)
    model.gradient_descent(1000)
    pred = model.make_predictions(test_set.drop(target_variable, axis=1).values.T)
    _, _, mse = eval_metrics(test_set[target_variable], pred)
    print('This is the MSE for the Autoencoder on the Machine data set: ', mse)


    data = cancer
    target_variable = 'class'
    pred_type = 'classification'
    test_set, train_set = class_split(data, 1, target_variable, False)

    model = Log_Reg(train_set, target_variable)
    model.fit(600, 0.01, 0.01)
    pred = model.predict(test_set.drop(target_variable, axis=1))
    acc = model.get_accuracy(pred, test_set[target_variable])
    print('\nThis is the Accuracy for the Logistic Regression on the Cancer data set: ', acc)

    model = Neural_Network(train_set, target_variable, pred_type, [19, 9], 0.001)
    model.gradient_descent(100)
    pred = model.make_predictions(test_set.drop(target_variable, axis=1).values.T)
    acc = model.get_accuracy(pred, test_set[target_variable])
    print('This is the Accuracy for the Feedforward Neural Network on the Cancer data set: ', acc)

    model = Auto_NN(train_set, target_variable, pred_type, [1, 5], 0.1)
    model.gradient_descent(100)
    pred = model.make_predictions(test_set.drop(target_variable, axis=1).values.T)
    acc = model.get_accuracy(pred, test_set[target_variable])
    print('This is the Accuracy for the Autoencoder on the Cancer data set: ', acc)



    # max_iter, alpha, num_node1, num_node2, mse = hyper_tune(val_set, target_variable, pred_type)
    # print(max_iter, alpha, num_node1, num_node2, mse)
    # # # max_iter = 1000
    # # # alpha = 0.0001
    # # # num_node1 = 1
    # # # num_node2 = 1
    # results = k2_cross(train_set, 1, target_variable, pred_type, max_iter, alpha, num_nodes=[num_node1, num_node2])
    # print(results)
    # model = Neural_Network(train_set, target_variable, 'regression')
    # model.gradient_descent(0.0001, 500)
    # X_test = val_set.drop(target_variable, axis=1).values.T
    # y_test = val_set[target_variable].values

    # print(w)
    # print(b)
    # print(yhat)

    # X_test = val_set.drop(target_variable, axis=1).values
    # y_test = val_set[target_variable].values
    # print(r2)
    # print(pearsons)

    #

    # #
    # val_set, train_set = class_split(data, 1, target_variable, True)
    #
    # Y = val_set[target_variable]
    # model = Log_Reg(train_set, target_variable)
    # model.fit(600, 0.01, 0.258)
    # # model.loss_plot()
    # # plt.show()
    # pred = model.predict(val_set.drop(target_variable, axis=1))
    # acc = model.get_accuracy(pred, Y)
    # # precision, recall, f1 = eval_metrics(Y, pred, 'classification')
    # print(acc)

    # model = Neural_Network(train_set, target_variable, 'classification', None)
    # model.gradient_descent(0.01, 500)
    # X_test = val_set.drop(target_variable, axis=1).values.T
    # y_test = val_set[target_variable].values
    # print(model.get_accuracy(model.make_predictions(X_test), y_test))

    # max_iter, alpha, num_node1, num_node2, accuracy = hyper_tune(val_set, target_variable, pred_type)
    # # # max_iter = 1000
    # # # alpha = 0.000001
    # # # num_node1 = 60
    # # # num_node2 = 20
    # # # # mu = 0.182
    # print(max_iter, alpha, num_node1, num_node2, accuracy)
    # results = k2_cross(train_set, 1, target_variable, 'classification', max_iter, alpha, num_nodes=[num_node1, num_node2])
    # print(results)

    # count = 1
    # for k in [1, 5, 10]:
    #     # for i in [5, 10, 15, 20, 25]:
    #     for j in [0.2, 0.02, 0.002, 0.0002, 0.00002]:
    #         # _, _, f1, _ = cross_val(folds, 'classification', 'log', eta=j, epoch=k)
    #         r2, mse, _ = cross_val(folds, 'regression', 'linear_reg', eta=j, epoch=k)
    #         sheet1.write(count, 1, k)
    #         # sheet1.write(count, 2, i)
    #         sheet1.write(count, 3, j)
    #         sheet1.write(count, 4, r2)
    #         count += 1
    #         print(count)
    # wb.save('tuning_abalone_lr.xls')
    # folds = k_folds(data, 'regression')
    # r2, mse, _ = cross_val(folds, 'regression', 'linear_reg')
    # print(r2, mse)
    # target = data.keys()[-1]
    # X = data.drop(target, axis=1)
    # y = data[target]
    # weights, bias, losses = logistic2(data, 10, 0.02)
    # # # print(weights)
    # # # pred = lin_predict(X, weights, bias)
    # pred = log_predict(X, weights, bias)
    # # # r2, mse = metrics(y, pred, 'regression')
    # # # print(r2, mse)
    # precision, recall, f1 = metrics(y, pred, 'classification')
    # print(f1)
    # print(log_accuracy(forest.keys()[-1], pred))

    # weights, bias, losses = linear(abalone, 100, 0.2)
    # print(weights)
    # print(weights)
    # print(bias)
    # print(losses)
    # print(log_accuracy(cars['class'], y_hat=losses))
    # Class = house.keys()[-1]
    # X = house.drop(Class, axis = 1)
    # y = house[Class]
    # w, b, l = logistic(X, y, 100, 1000, 0.2)
    #
    # print(log_accuracy(X, log_predict(X, w, b)))
