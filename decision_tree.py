"""This module includes methods for training and predicting using decision trees."""
import numpy as np
from pprint import pprint



def calculate_information_gain(data, labels):
    """
    Computes the information gain on label probability for each feature in data

    :param data: d x n matrix of d features and n examples
    :type data: ndarray
    :param labels: n x 1 vector of class labels for n examples
    :type labels: array
    :return: d x 1 vector of information gain for each feature (H(y) - H(y|x_d))
    :rtype: array
    """
    all_labels = np.unique(labels)
    # Find what labels are there in my dataset
    num_classes = len(all_labels)
    # Number of classes
    class_count = np.zeros(num_classes)

    n: object
    d, n = data.shape

    # first step Calculate the entropy
    full_entropy = 0
    for c in range(num_classes):
        class_count[c] = np.sum(labels == all_labels[c])
        if class_count[c] > 0:
            class_prob = class_count[c] / n
            # class_prob is the portion of the dataset that have class c [pi c]
            full_entropy -= class_prob * np.log(class_prob)
            #full_entropy= H(pi)

     #print("Full entropy is %d\n"%full_entropy)


    gain = full_entropy * np.ones(d)
    #initial the Information gain???

    # we use a matrix dot product to sum to make it more compatible with sparse matrices
    num_x = data.dot(np.ones(n))
    prob_x = num_x / n
    prob_not_x = 1 - prob_x

    for c in range(num_classes):
        # print("Computing contribution of class %d." % c)
        num_y = np.sum(labels == all_labels[c])
        # this next line sums across the rows of data, multiplied by the
        # indicator of whether each column's label is c. It counts the number
        # of times each feature is on among examples with label c.
        # We again use the dot product for sparse-matrix compatibility
        data_with_label = data[:, labels == all_labels[c]]
        num_y_and_x = data_with_label.dot(np.ones(data_with_label.shape[1]))

        # Prevents Python from outputting a divide-by-zero warning
        with np.errstate(invalid='ignore'):
            prob_y_given_x = num_y_and_x / (num_x + 1e-8)
        prob_y_given_x[num_x == 0] = 0

        nonzero_entries = prob_y_given_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                cond_entropy = - np.multiply(np.multiply(prob_x, prob_y_given_x), np.log(prob_y_given_x))
            gain[nonzero_entries] -= cond_entropy[nonzero_entries]

        # The next lines compute the probability of y being c given x = 0 by
        # subtracting the quantities we've already counted
        # num_y - num_y_and_x is the number of examples with label y that
        # don't have each feature, and n - num_x is the number of examples
        # that don't have each feature
        with np.errstate(invalid='ignore'):
            prob_y_given_not_x = (num_y - num_y_and_x) / ((n - num_x) + 1e-8)
        prob_y_given_not_x[n - num_x == 0] = 0

        nonzero_entries = prob_y_given_not_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                cond_entropy = - np.multiply(np.multiply(prob_not_x, prob_y_given_not_x), np.log(prob_y_given_not_x))
            gain[nonzero_entries] -= cond_entropy[nonzero_entries]

    return gain


def decision_tree_train(train_data, train_labels, params):
    """Train a decision tree to classify data using the entropy decision criterion.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include a 'max_depth' value
    :type params: dict
    :return: dictionary encoding the learned decision tree
    :rtype: dict
    """
    max_depth = params['max_depth']

    labels = np.unique(train_labels)
    num_classes = labels.size


    model = recursive_tree_train(train_data, train_labels, depth=0, max_depth=max_depth, num_classes=num_classes)
    pprint(model)
    return model


def recursive_tree_train(data, labels, depth, max_depth, num_classes):
    """Helper function to recursively build a decision tree by splitting the data by a feature.
    [0, 1, 0, 1]

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param labels: length n numpy array with integer labels
    :type labels: array_like
    :param depth: current depth of the decision tree node being constructed
    :type depth: int
    :param max_depth: maximum depth to expand the decision tree to
    :type max_depth: int
    :param num_classes: number of classes in the classification problem (20)
    :type numclasses: int
    :return: dictionary encoding the learned decision tree node
    :rtype: dict
    """
    # TODO: INSERT YOUR CODE FOR LEARNING THE DECISION TREE STRUCTURE HERE
    #node= {max_Info_gain: { }}
    #print("data shape", data.shape)
    #print("data", data)
    node ={}
    #dictionary
    count= np.array([np.zeros(num_classes)])
    #print("labels", labels)

    for i in labels:
        for j in range(len(labels)):
            if i == j:
                count[0,j]+=1
    #print('n',len(labels))

    #print("count", count)
    #print('sum_count',count.sum())
    y = np.argmax(count)
    #print("depth, de, max_d, label",depth, max_depth, np.unique(labels))
    if depth == max_depth:
         node['predicted'] = y
         return node
    elif len(np.unique(labels))==1:
        node['predicted'] = y
        return node
    else:
        gain= calculate_information_gain(data, labels)

        max_Info_gain = np.argmax(gain)
        index_left = np.nonzero(data[max_Info_gain])
        #when attribute i = 1
        #print("index", index_left)
        left= data[:,index_left]
        left_labels= labels[index_left]
        # data = d*n
        left = np.squeeze(left)
        #print('left',left)
        #subdataset
        #print(left.shape)

        index_right = np.nonzero(1- data[max_Info_gain])
        # when attribute i = 0
        right = data[:,index_right]
        right_labels = labels[index_right]

        right = np.squeeze(right)
        #print('right', right)
        #print(right.shape)

        #print("depth", depth)

        node_left = recursive_tree_train(left,left_labels,depth+1 ,max_depth,num_classes)
        node_right = recursive_tree_train(right,right_labels,depth+1,max_depth,num_classes)

        node[f'left {max_Info_gain}'] = node_left
        node[f'right {max_Info_gain}'] = node_right
        return node

def predict_point(point, model):
    if 'predicted' in model.keys():
        label = model['predicted']
        #print('predicted')
        #print('label_point',label)
        return label
    else:
        branch_x = list(model.keys())
        max_info_gain_train = int(branch_x[0].split()[1])
        #print('branching')
        if point[max_info_gain_train] > 0:
            return predict_point(point, model['left '+ str(max_info_gain_train)])

        elif point[max_info_gain_train] == 0:
            return predict_point(point, model['right '+ str(max_info_gain_train)])

# def predict_points(data, model):
#     labels = []
#     for point in data.T:
#         label = predict_point(point,model)
#         #print('label', label)
#         labels.append(label)
#
#     return labels

def decision_tree_predict(data, model):
    """Predict most likely label given computed decision tree in model.

    :param data: d x n ndarray of d binary features for n examples.
    :type data: ndarray
    :param model: learned decision tree model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    labels = []
    for point in data.T:
        label = predict_point(point, model)
        # print('label', label)
        labels.append(label)
    return labels
    # TODO: INSERT YOUR CODE FOR COMPUTING THE DECISION TREE PREDICTIONS HERE
    # labels = {}
    # # dictionary
    # if model['predicted'] != 0:
    #     labels = model['predicted']
    # else:
    #
    #     branch_x = list(model.keys())
    #     max_info_gain_train =int(branch_x[0].split()[1])
    #     index_left = np.nonzero(data[max_info_gain_train])
    #     #  what  to  put here
    #     # when attribute i = 1
    #     # print("index", index_left)
    #     left = data[:, index_left]
    #     left_labels = labels[index_left]
    #
    #
    #     #left_labels = decision_tree_predict(left, model['left'])
    #     # data = d*n
    #     left = np.squeeze(left)
    #     # print('left',left)
    #     # subdataset
    #     # print(left.shape)
    #
    #     index_right = np.nonzero(1 - data[max_info_gain_train])
    #     right = data[:, index_right]
    #     #right_labels = decision_tree_predict(right, model['right'])
    #     right_labels = labels[index_right]
    #     right = np.squeeze(right)
    #     # print('right', right)
    #     # print(right.shape)
    #
    #     model_left = decision_tree_predict(left, model['left'])
    #     model_right = decision_tree_predict(right, model['right'])
    #     #model_left = decision_tree_predict (left, left_labels, depth + 1, max_depth, num_classes)
    #     #model_right = decision_tree_predict(right, right_labels, depth + 1, max_depth, num_classes)
    #
    #     labels[index_left] = left_labels
    #     labels[index_right] = right_labels

