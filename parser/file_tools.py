import os
import json
import pickle
import gzip
import random
import pandas as pd

from Bio import SeqIO
from sklearn.model_selection import train_test_split

aa_set = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
              'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y'}


def check_seq_list(path):
    with open(f"data/{path}", "rb") as input_file:
        already_exist = pickle.load(input_file)
    print(len(list(set(already_exist))))
    print('len 0', len(already_exist[0]))
    print('0', already_exist[0])
    print('len -1', len(already_exist[-1]))
    print('-1', already_exist[-1])


def select_file(path):
    select_option = 'Выберете файл:\n'
    check_files_path = os.path.join('data/', path)
    files = os.listdir(check_files_path)
    option: dict[int, str] = {}
    for fid, fname in enumerate(files):
        select_option += f'{fid + 1}: {fname}\n'
        option[fid + 1] = fname
    choice = input(select_option)
    fname = option[int(choice)] if choice.isnumeric() else None
    if fname is None:
        print('Выберете один из вариантов, перечисленных в списке')
        select_file()
    return fname


def select_folder():
    """
    Usefull only for my progect setup
    :return:
    """
    path = ''
    print('select folder')
    print('1. sequential')
    print('2. parallel')
    print('3. seq_40_more')
    print('4. seq_40_less')
    folder_choise = input()
    if int(folder_choise) == 1:
        path += 'pkl_from_parser/'
        path += select_file(path)
        check_seq_list(path)
    elif int(folder_choise) == 2:
        path += 'pkl_from_parser_prl/'
        path += select_file(path)
        check_seq_list(path)
    elif int(folder_choise) == 3:
        with open(r"data/seq_40_more.pkl", "rb") as input_file:
            already_exist = pickle.load(input_file)
        print(len(list(set(already_exist))))
        print('len 0', len(already_exist[0]))
        print('0', already_exist[0])
        print('len -1', len(already_exist[-1]))
        print('-1', already_exist[-1])
    elif int(folder_choise) == 4:
        path += 'pkl_from_parser_prl_5_40/'
        path += select_file(path)
        check_seq_list(path)
    else:
        print('no option exists')
        return
    return


def cut_before_exact_len_with_stat(seq_list, len_: int):
    """
    Cut list of prot sequences till exact len. Return also numerical statistics of number of eche len.
    :param seq_list: list of seq
    :param len_: exact len
    :return: None
    """
    len_dict = {}
    for i in range(1, len_, 1):
        len_dict[str(i)] = 0

    before_len_list = []

    for seq in seq_list:
        s_ln = len(seq)
        if s_ln < len_ and set(seq).issubset(aa_set):
            len_dict[str(s_ln)] += 1
            before_len_list.append(seq)

    with open(f"seq_{len_}_less.pkl", 'wb') as f:
        pickle.dump(before_len_list, f)
    with open('len_dict.json', 'w') as fp:
        json.dump(len_dict, fp)


def len_statistics_from_len_in_filename(folder_list):
    """
    If sequences was parsed from func, their names consist numbers of samples and len of sequence. This func aggregates
    this information and save it to json.
    :param folder_list: list of str paths to list files.
    :return: None
    """
    len_dict = {}
    for folder in folder_list:
        files = os.listdir(folder)
        for file_name in files:
            name_split = file_name.split('_')
            real_len = name_split[2].split('.')
            len_dict[name_split[1]] = int(real_len[0])

    with open('len_dict_fn.json', 'w') as fp:
        json.dump(len_dict, fp)


def open_pkl_files(folder_name, file_name):
    path = os.path.join('data', folder_name, file_name)
    with open(f"{path}", "rb") as input_file:
        one_len_list = pickle.load(input_file)
    return one_len_list


def read_all_from_dir(folder):
    full_sequence_in_folder = []
    check_files_path = os.path.join('data', folder)
    files = os.listdir(check_files_path)
    for file in files:
        full_one_len_list = open_pkl_files(folder, file)
        full_sequence_in_folder.append(full_one_len_list)
    return full_sequence_in_folder


def clay_several_files():
    with open(f"rna_10.pkl", "rb") as input_file:
        dna_5 = pickle.load(input_file)

    with open(f"rna_11.pkl", "rb") as input_file:
        dna_6 = pickle.load(input_file)

    with open(f"rna_12.pkl", "rb") as input_file:
        dna_7 = pickle.load(input_file)

    with open(f"rna_13.pkl", "rb") as input_file:
        dna_8 = pickle.load(input_file)
    seq_96 = dna_5 + dna_6 + dna_7 + dna_8

    with open(f"data/rna_10_till_13.pkl", 'wb') as f:
        pickle.dump(seq_96, f)


def clay_files_one_len(len_):
    seq_one_len_list = []
    folder_name = 'pkl_from_parser_prl_5_40'
    check_files_path = os.path.join('data', folder_name)
    files = os.listdir(check_files_path)
    list_to_save = []
    for file in files:
        if file.startswith(f"seq_{len_}"):
            seq_one_len_list.append(file)
    print(seq_one_len_list)
    for file_name in seq_one_len_list:
        list_to_append = open_pkl_files(folder_name, file_name)
        print(len(list_to_append))
        for seq in list_to_append:
            if seq not in list_to_save:
                list_to_save.append(seq)
    with open(f"data/pkl_from_parser_prl_5_40/seq_{len_}_{len(list_to_save)}.pkl", 'wb') as f:
        pickle.dump(list_to_save, f)


def make_pickle_files(
    sequence_files_name: list,
    ready_list: list,
    known_aa: set = {'A', 'C', 'G', 'T'}
):
    for file_name in sequence_files_name:
        one_len_seq_list = []
        fasta_sequences = SeqIO.parse(open(file_name), 'fasta')

        for fasta in fasta_sequences:
            sequence = str(fasta.seq)
            if sequence not in ready_list and set(sequence).issubset(known_aa):
                one_len_seq_list.append(sequence)

        one_len_seq_list_unique = list(set(one_len_seq_list))
        print(file_name, len(one_len_seq_list_unique))
        ready_list.extend(one_len_seq_list_unique)
        print(len(ready_list))

    with open("seq_40_96_with_bact.pkl", 'wb') as f:
        pickle.dump(ready_list, f)


def read_fasta_file(file_path):
    # sequences_10 = []
    # sequences_11 = []
    # sequences_12 = []
    sequences_14 = []
    with gzip.open("length14_TO_14_AND_entry_typeSequence.fasta.gz", "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
                seq = str(record.seq)
                if set(seq).issubset(aa_set):
                    if len(seq) == 14 and seq not in sequences_14:
                        if len(sequences_14) >= 1200:
                            break
                        sequences_14.append(seq)
                    # elif len(seq) == 11 and seq not in sequences_11:
                    #     if len(sequences_11) > 1200:
                    #         continue
                    #     sequences_11.append(seq)
                    # elif len(seq) == 12 and seq not in sequences_12:
                    #     if len(sequences_12) > 1200:
                    #         continue
                    #     sequences_12.append(seq)
                    # elif len(seq) == 13 and seq not in sequences_13:
                    #     if len(sequences_13) > 1200:
                    #         continue
                    #     sequences_13.append(seq)
    print(len(sequences_14))
    with open("rna_14.pkl", 'wb') as f:
        pickle.dump(sequences_14, f)


def sts_clay(folders):
    """
    Takes files from folders, takes exact number of sequences randomly from the file and makes train and test sets.
    :param folders:
    :return:
    """
    test = []
    train = []
    for folder in folders:
        check_files_path = os.path.join('data', folder)
        files = os.listdir(check_files_path)
        for file in files:
            full_one_len_list = open_pkl_files(folder, file)
            test_one_len = random.sample(full_one_len_list, 360) # exact num
            for_train_one_len = list(set(full_one_len_list) - set(test_one_len))
            train_one_len = random.sample(for_train_one_len, 840) # exact num
            print(f"for len={len(train_one_len[0])} test_len = {len(test_one_len)}, train_len = {len(train_one_len)}")
            test.extend(test_one_len)
            train.extend(train_one_len)
    return train, test


def aptamer_sts_clay(folders):
    """
    Takes exact num of sequences from folder parsed from ncbi
    :param folders:
    :return:
    """
    prot = []
    for folder in folders:
        check_files_path = os.path.join('data', folder)
        files = os.listdir(check_files_path)
        for file in files:
            full_one_len_list = open_pkl_files(folder, file)
            test_one_len = random.sample(full_one_len_list, 1200) # exact num
            print(f"for len={len(test_one_len[0])} all = {len(test_one_len)}")
            prot.extend(test_one_len)
    return prot


def str_from_files(files):
    """
    Makes train and test as 3/7, stratified by length by processing files of the same length sequentially
    :param files:
    :return:
    """
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


def traintest_making():
    """
    Makes balanced (aa, len) train test from pickle, picks statistics of amino acids and len.
    :return:
    """
    with open(r"data/seq_96_all.pkl", "rb") as input_file:
        all_seqs_full = pickle.load(input_file)
    df = pd.DataFrame()
    df['s'] = all_seqs_full

    df['len'] = df['s'].str.len()
    for i in aa_set:
        df[i] = df['s'].str.count(i)

    for rand_st in range(150):
        train, test = train_test_split(df, shuffle=True, test_size=0.3, random_state=rand_st)
        train_distr = train.describe()
        train_distr_mean = train_distr.loc['mean']

        test_distr = test.describe()
        test_distr_mean = test_distr.loc['mean']

        result_minus = abs(test_distr_mean-train_distr_mean)
        aa_distribution = result_minus.drop('len')
        if test_distr.loc['max', 'len'] > train_distr.loc['max', 'len']:
            continue
        elif aa_distribution[aa_distribution > 0.1].shape[0] > 0:
            continue
        elif result_minus['len'] > 0.5:
            continue
        else:
            break
    test_distr.to_csv(f"test_distr_{rand_st}.csv")
    train_distr.to_csv(f"train_distr_{rand_st}.csv")

    train_seq_ = train['s'].to_list()
    with open('train_seq_clean.pkl', 'wb') as f:
        pickle.dump(train_seq_, f)
    del train_seq_

    test_seq_ = test['s'].to_list()
    with open('test_seq_clean_.pkl', 'wb') as f:
        pickle.dump(test_seq_, f)
