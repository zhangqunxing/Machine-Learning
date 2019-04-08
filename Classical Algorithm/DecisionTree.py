#FileName: DecisionTree
#Author: zhangqx
#Date: 2018-5-12

import numpy as np
import sklearn as skl
import os
from math import log
import operator


def create_data():
    data_set=[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['label_A', 'label_B']
    return data_set, labels

def caculate_entropy(data_set):
    """
    To caculate the entropy of the data_set
    :param data_set: two dimentional arrays
    :return: entropy of data_set
    """
    #To count rows of data_set
    sample_num = len(data_set)

    label_counts = {}
    for each_feature in data_set:
        current_label = each_feature[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 1
        else:
            label_counts[current_label] +=1

    #To caculate the entropy
    entropy = 0.0;
    for key in label_counts:
        prob = float(label_counts[key]/sample_num)
        entropy += -prob*log(prob,2)
    return entropy;

def split_data_set(data_set, index, value):
    """
    To split the data_set to subset according to the value which is given;
    :param data_set:  two dimentional arrays
    :param axis:      index of eigenvalue
    :param value:     eigenvalue
    :return:          subset
    """
    sub_data_set =[]
    for each_data in  data_set:
        if each_data[index] == value:
            sub_feature = each_data[:index]
            sub_feature.extend(each_data[index+1:])
            sub_data_set.append(sub_feature)
    return sub_data_set

def choose_best_feature_to_split(data_set):
    """
    To choose the best feature according to IE3 or c4.5
    :param data_set: two dimentional arrays
    :return: index of the best feature
    """
    #number of feature
    num_feature = len(data_set[0]) -1

    #H(D)
    base_entropy = caculate_entropy(data_set)
    best_info_gain = 0.0
    best_feature_index = -1

    for feature_index in range(num_feature):
        #Get all values of a feature
        feature_list = [each_data[feature_index] for each_data in data_set]

        #Get unique feature list from feature_list
        unique_feature_list = set(feature_list)

        entropy_feature = 0.0;
        # *************************************************
        # C4.5 caculate GainRatio = info_gain/intrinsic_info
        intrinsic_info = 0.0;
        # **************************************************
        for each_feature_value in unique_feature_list:
            sub_data_set = split_data_set(data_set, feature_index, each_feature_value)

            #to caculate the sum of entropy to sub_data_set
            prob = len(sub_data_set)/float(len(data_set))
            entropy_feature += prob * caculate_entropy(sub_data_set)

            # **************************************************
            #to caculate intrinsicInfo
            intrinsic_info += -prob*log(prob, 2);   #Pay attention to '-'
            # **************************************************

        #caculate info_gain
        info_gain = base_entropy - entropy_feature

        # *************************************************
        # C4.5 caculate GainRatio = info_gain/intrinsic_info
        if intrinsic_info != 0.0 :
            GainRatio = info_gain/intrinsic_info;
            info_gain = GainRatio;
        # *************************************************


        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature_index = feature_index

        return best_feature_index

def count_majority_class(class_list):
    """
    To count majority feature of list
    :param class_list: list
    :return: label
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote]=1
        else:
            class_count[vote]+=1
    sorted_class_count = sorted(class_count.items(), key = lambda item:item[1], reverse = True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    consturct a decision tree
    :param data_set:  two dimentional arrays
    :param labels:   label_set
    :return: decision tree
    """
    class_list = [sample[-1] for sample in data_set]   #['yes', 'yes', 'no', 'no','no']

    #if it is a same label, to reture
    if class_list.count(class_list[-1]) == len(class_list):
        return class_list[-1]

    #if feature class is only one， return the most class
    if len(data_set[0]) ==1:
        return count_majority_class(class_list)
    #create
    best_feature_index = choose_best_feature_to_split(data_set)
    best_feature_label = labels[best_feature_index]

    decision_tree = {best_feature_label:{}}
    del(labels[best_feature_index])

    best_feature_value = [sample[best_feature_index] for sample in data_set]
    unique_feature_value = set(best_feature_value)

    for feature_value in unique_feature_value:
        sub_labels = labels[:]

        #recurrently create decision tree
        sub_data_set = split_data_set(data_set, best_feature_index, feature_value)
        decision_tree[best_feature_label][feature_value] = create_tree(sub_data_set,sub_labels)

    return decision_tree

def caculate_gini(data_set):
    """
    To caculate the gini param of the data_set
    :param data_set:
    :return: gini param
    """
    label_value = [each[-1] for each in data_set]
    num_label = len(label_value)

    count_label={}
    for each_label in label_value:
        if each_label in count_label.keys():
            count_label[each_label] +=1
        else:
            count_label[each_label] =1

    count = 0.0
    for key in count_label.keys():
        count += (count_label[key]/num_label)**2
    return 1- count

def classify(train_decision_tree, feature_labels, test_vec):
    """
    To predict test dataset according to the trained decision tree 
    :param train_decision_tree:
    :param feature_labels:
    :param test_vec:
    :return:
    """
    first_feature = list(train_decision_tree.keys())[0]
    feature_value_dict = train_decision_tree(first_feature)

    first_feature_index = feature_labels.index(first_feature)

    for each in feature_value_dict.keys():
        if each == test_vec[first_feature_index]:
            if type(feature_value_dict[key]).__name__ == 'dict':
                return classify(feature_value_dict[key], feature_labels, test_vec)
            else:
                return feature_value_dict[key]

def test_function():
	"""
	To test each function
	"""
    data, target = create_data()
    print(caculate_entropy(data))
    # test split_data_set
    print(split_data_set(data, 1, 1))

    test_smaple = ['yes', 'yes', 'no', 'no','no']
    print(count_majority_class(test_smaple))

    print(create_tree(data, target))

if __name__ == '__main__':
    test_function()
