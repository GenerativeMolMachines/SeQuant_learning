import pickle
import os
from pickle_clay import open_pkl_files
import random


def sts_clay(folders):
    test = []
    train = []
    for folder in folders:
        check_files_path = os.path.join('data', folder)
        files = os.listdir(check_files_path)
        for file in files:
            full_one_len_list = open_pkl_files(folder, file)
            n = int(len(full_one_len_list)*0.3)
            test_one_len = random.sample(full_one_len_list, n)
            train_one_len = list(set(full_one_len_list) - set(test_one_len))
            print(f"for len={len(train_one_len[0])} test_len = {len(test_one_len)}, train_len = {len(train_one_len)}")
            test.extend(test_one_len)
            train.extend(train_one_len)
    return train, test


if __name__ == "__main__":
    train_l, test_l = sts_clay(['pkl_from_parser_prl_5_40', 'pkl_from_parser', 'pkl_from_parser_prl'])
    with open(f"data/train_seq_clean_str.pkl", 'wb') as f:
        pickle.dump(train_l, f)

    with open(f"data/test_seq_clean_str.pkl", 'wb') as f:
        pickle.dump(test_l, f)