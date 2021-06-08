import os

import numpy as np
import pandas as pd
import sklearn.utils


def split_dataset(file_in, test_split: int = 0.2, validate_split: int = 0.1):
    """ split one labels file into train, test, validate labels files. Creates the following files: train_labels.json, test.json, validate.json in the same folder as the source file is located

    Args:
        file_in: path to labels file
        test_split: test percentage
        validate_split: validate percentage
    """

    base_path = os.path.dirname(file_in)
    train_out_filename = 'train.txt'
    test_out_filename = 'test.txt'
    validate_out_filename = 'validate.txt'

    # data_df = pd.read_csv(os.path.join(os.getcwd(), 'driving_dataset', 'data.txt'), index_col=False)
    data_df = np.loadtxt(os.path.join(os.getcwd(), 'driving_dataset', 'data.txt'),dtype='str')
    data_df = sklearn.utils.shuffle(data_df, random_state=0)

    last_train_index = int((1-test_split-validate_split)*len(data_df))
    last_test_index = int((1-validate_split)*len(data_df))

    train_list = data_df[:last_train_index]
    test_list = data_df[last_train_index:last_test_index]
    validate_list = data_df[last_test_index:]

    np.savetxt(os.path.join(base_path, train_out_filename), train_list,fmt='%s')
    np.savetxt(os.path.join(base_path, test_out_filename), test_list,fmt='%s')
    np.savetxt(os.path.join(base_path, validate_out_filename), validate_list,fmt='%s')




if __name__ == '__main__':
    split_dataset("driving_dataset/data.txt", 0.2, 0.1)
