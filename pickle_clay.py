import pickle
import os


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


def clay_two_files():
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

if __name__ == "__main__":
    # for i in [17, 18, 19, 20, 21, 38]:
    #     clay_files_one_len(i)
    # before_60 = read_all_from_dir('pkl_from_parser_prl_5_40')
    # print('len(here):', len(before_60))
    # # more_60 = read_all_from_dir('pkl_from_parser_prl')
    # # print('len(more_60):', len(more_60))
    # # more_40_list = before_60 + more_60
    # # print('len(all):', len(more_40_list))
    # with open(f"data/seq_40_less.pkl", 'wb') as f:
    #     pickle.dump(before_60, f)
    # print('ok')
    open_pkl_files('', 'prot_all.pkl')
    # clay_two_files()