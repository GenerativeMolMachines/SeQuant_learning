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
    with open(f"data/seq_40_less.pkl", "rb") as input_file:
        seq_40_less = pickle.load(input_file)

    with open(f"data/seq_40_more.pkl", "rb") as input_file:
        seq_40_more = pickle.load(input_file)
    seq_96 = seq_40_less + seq_40_more
    with open(f"data/seq_96.pkl", 'wb') as f:
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
    for file_name in seq_one_len_list:
        list_to_append = open_pkl_files(folder_name, file_name)
        for seq in list_to_append:
            if seq not in list_to_save:
                list_to_save.append(seq)
    with open(f"data/pkl_from_parser_prl_5_40/seq_{len_}_{len(list_to_save)}.pkl", 'wb') as f:
        pickle.dump(list_to_save, f)

if __name__ == "__main__":
    clay_files_one_len(12)
    # before_60 = read_all_from_dir('pkl_from_parser')
    # print('len(before_60):', len(before_60))
    # more_60 = read_all_from_dir('pkl_from_parser_prl')
    # print('len(more_60):', len(more_60))
    # more_40_list = before_60 + more_60
    # print('len(all):', len(more_40_list))
    # with open(f"data/seq_40_more.pkl", 'wb') as f:
    #     pickle.dump(more_40_list, f)
    # print('ok')

    # clay_two_files()
