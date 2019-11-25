import pandas as pd
import numpy as np


def get_train_test_idxs(df, train_frac=0.8):
    dataset_size = len(df)
    train_set_size = int(train_frac * dataset_size)
    rand_perm = np.random.permutation(dataset_size)
    train_idxs = rand_perm[:train_set_size]
    test_idxs = rand_perm[train_set_size:]
    return train_idxs, test_idxs


def add_set_column_and_save(csv_path):
    df = pd.read_csv(csv_path)
    clean_df = df[df['Quantity'] == 1]
    train_idxs, test_idxs = get_train_test_idxs(clean_df)
    set_list = np.zeros(len(clean_df))
    set_list[test_idxs] = 1
    clean_df['set'] = set_list
    clean_df.to_csv(csv_path)


if __name__ == '__main__':
    add_set_column_and_save('/media/ron/Data/google_time/repos/pets_adopt/data/train.csv')
