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
            test_one_len = random.sample(full_one_len_list, 360)
            for_train_one_len = list(set(full_one_len_list) - set(test_one_len))
            train_one_len = random.sample(for_train_one_len, 840)
            print(f"for len={len(train_one_len[0])} test_len = {len(test_one_len)}, train_len = {len(train_one_len)}")
            test.extend(test_one_len)
            train.extend(train_one_len)
    return train, test

def aptamer_sts_clay(folders):
    prot = []
    for folder in folders:
        check_files_path = os.path.join('data', folder)
        files = os.listdir(check_files_path)
        for file in files:
            full_one_len_list = open_pkl_files(folder, file)
            test_one_len = random.sample(full_one_len_list, 1200)
            print(f"for len={len(test_one_len[0])} all = {len(test_one_len)}")
            prot.extend(test_one_len)
    return prot


def str_from_files(files):
    test = []
    train = []
    for file in files:
        path = os.path.join(file)
        with open(f"{path}", "rb") as input_file:
            one_len_list = pickle.load(input_file)
        test_one_len = random.sample(one_len_list, int(len(one_len_list)*0.3))
        train_one_len = list(set(one_len_list) - set(test_one_len))
        print(f"for len={len(train_one_len[0])} test_len = {len(test_one_len)}, train_len = {len(train_one_len)}")
        test.extend(test_one_len)
        train.extend(train_one_len)
    return train, test

if __name__ == "__main__":
    prot_all = aptamer_sts_clay(['pkl_from_parser', 'pkl_from_parser_prl', 'pkl_from_parser_prl_5_40'])

    # train_l, test_l = str_from_files(['rna_10.pkl', 'rna_11.pkl', 'rna_12.pkl', 'rna_13.pkl',
    #                                   'dna_5.pkl', 'dna_6.pkl', 'dna_7.pkl', 'dna_8.pkl'])
    # with open(f"test_prot_for_masha.pkl", "rb") as input_file:
    #     test_prot = pickle.load(input_file)
    # with open(f"train_prot_for_masha.pkl", "rb") as input_file:
    #     train_prot = pickle.load(input_file)
    #
    # train_l.extend(train_prot)
    # test_l.extend(test_prot)
    #
    with open(f"prot_all_with_88_90.pkl", 'wb') as f:
        pickle.dump(prot_all, f)
    #
    # with open(f"train_prot_full_dna_10_till_13_rna_5_till_8.pkl", 'wb') as f:
    #     pickle.dump(train_l, f)