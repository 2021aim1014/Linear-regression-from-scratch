import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def standardize_list(data_list):
    mean = float(sum(data_list)) / len(data_list)
    temp_list = [(each - mean)**2 for each in data_list]
    sd = math.sqrt(sum(temp_list)/len(temp_list))
    data_list = [(each - mean)/sd for each in data_list]

    # mean = float(sum(data_list)) / len(data_list)
    # temp_list = [(each - mean)**2 for each in data_list]
    # sd = math.sqrt(sum(temp_list)/len(temp_list))
    # print(mean)
    # print(sd)
    return data_list

def standardize_data(data):
    age_list = data['age'].tolist()
    data['age'] = pd.DataFrame(standardize_list(age_list))

    bmi_list = data['bmi'].tolist()
    data['bmi'] = pd.DataFrame(standardize_list(bmi_list))
    #
    children_list = data['children'].tolist()
    data['children'] = pd.DataFrame(standardize_list(children_list))

    charges_list = data['charges'].tolist()
    data['charges'] = pd.DataFrame(standardize_list(age_list))

def split_data_into_train_test(data):
    N = 10
    K_fold_data = []
    training_data = data.sample(frac=0.8, random_state=25)
    testing_data = data.drop(training_data.index)
    partition_size = training_data.shape[0] // N
    for i in range(N-1):
        temp = training_data.sample(partition_size)
        training_data = training_data.drop(temp.index)
        K_fold_data.append(temp)
    K_fold_data.append(training_data)
    return K_fold_data, testing_data

def compute_loss(train_x, train_y, weights, lambda_p):
    Y = np.dot(train_x, np.transpose(weights))
    JW = (1/2) * np.mean((train_y - Y) ** 2 )
    loss = JW + (1/2) * lambda_p * sum(weights[0, :]**2)
    return loss

def convert_to_numeric(col):
    col['sex'] = sex_mapping[col['sex']]
    col['region'] = region_mapping[col['region']]
    col['smoker'] = smoker_mapping[col['smoker']]
    return col

def gradients(train_x, train_y, weights, lambda_p):
    Y = np.dot(train_x, np.transpose(weights))
    N = train_x.shape[0]
    F = train_x.shape[1]
    grad = []
    for i in range(F):
        temp = np.mean((Y - train_y) * train_x[:, i])
        temp +=  lambda_p * weights[0, :][i]
        grad.append(temp)
    return(grad)

def ridgereg(train_x, train_y, lambda_p):
    eta = 0.01    # leaning rate
    weights = np.random.rand(1, train_x.shape[1])
    for i in range(100):
        grad = gradients(train_x, train_y, weights, lambda_p)
        grad = [each*eta for each in grad]
        weights = weights - grad
    return weights

data = pd.read_csv("insurance.csv")
standardize_data(data)

sex_unique_data = data['sex'].unique()
region_unique_data = data['region'].unique()
smoker_unique_data = data['smoker'].unique()
sex_mapping = dict([(v, i) for i, v in enumerate(sex_unique_data)])
region_mapping = dict([(v, i) for i, v in enumerate(region_unique_data)])
smoker_mapping = dict([(v, i) for i, v in enumerate(smoker_unique_data)])
data = data.apply(convert_to_numeric, axis=1)

training_data_k, testing_data = split_data_into_train_test(data)

N = 10
loss_list = []
_lambdas = [0.001, 0.01, 1 , 10]
for lambda_p in _lambdas:
    # print('')
    # print('Starting Linear Ridge Regression')
    weights_list = []
    # best_lambda = _lambdas[0]
    # best_loss = 10000
    for k in range(N):
        train_x = []
        train_y = []
        validate_x = training_data_k[k][['age', 'sex', 'bmi', 'children', 'smoker', 'region']].to_numpy()
        validate_y = training_data_k[k]['charges'].to_numpy()
        for i in range(10):
            if i != k:
                temp1 = training_data_k[i]
                temp = temp1[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].to_numpy()
                train_x.append(temp)
                temp = temp1[['charges']].to_numpy()
                train_y.append(temp)

        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y)
        weights = ridgereg(train_x, train_y, lambda_p)
        weights_list.append(weights)
        loss_on_validation = compute_loss(validate_x, validate_y, weights, lambda_p)
        # if(loss_on_validation < best_loss):
        #     loss_on_validation = best_loss
        #     best_lambda = lambda_p
        # print('')
        # print("Lambda value = ", lambda_p)
        # print("Validation partition = ", k)
        # print("Loss = ", loss_on_validation)

    weights_list = np.array(weights_list)
    weights = np.mean(weights_list, axis=1)
    train_x = testing_data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].to_numpy()
    train_y = testing_data[['charges']].to_numpy()
    loss = compute_loss(train_x, train_y, weights, lambda_p)
    # print('')
    # print("Lambda value = ", lambda_p)
    # print("MINIMUM LOSS = ", loss)
    loss_list.append(loss)

# print(_lambdas)
# print(loss_list)
title = 'Linear regression'
plt.title(title)
plt.xlabel('Lambda Values')
plt.ylabel('Loss')
# x = [0.001, 0.01, 1, 10]
# y = [0.5103609949557398, 0.4133446179462198, 0.5170026538968758, 0.502746003736156]
plt.plot(_lambdas, loss_list, 'r^-')
plt.show()

'''
OBSERVATION:

For lambda value = 0.01, the loss calculate is least

'''
