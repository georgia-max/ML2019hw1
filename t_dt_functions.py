import numpy as np
from decision_tree import decision_tree_train


def unique_data():
    """Set up uniquely identifiable examples"""
    train_data = np.zeros((4, 10))

    # label = 0
    train_data[:, 0] = [0, 0, 0, 0]
    train_data[:, 1] = [0, 0, 0, 1]
    train_data[:, 2] = [0, 0, 1, 0]
    train_data[:, 3] = [0, 0, 1, 1]
    train_data[:, 4] = [0, 1, 0, 0]

    # label = 1
    train_data[:, 5] = [0, 1, 0, 1]
    train_data[:, 6] = [0, 1, 1, 0]
    train_data[:, 7] = [0, 1, 1, 1]
    train_data[:, 8] = [1, 0, 0, 0]
    train_data[:, 9] = [1, 0, 0, 1]

    train_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    return train_data, train_labels

if __name__=="__main__":
    train_data, train_labels = unique_data()
    # train_data, train_labels = self.real_data()

    # set tree depth to unlimited
    params = {"max_depth": np.inf}

    model = decision_tree_train(train_data, train_labels, params)
    print('this is the model', model)
    print(type(model))