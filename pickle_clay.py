import pickle
import os


def clay_pkl_files(folder_name, file_name):
    path = os.path.join('data/', folder_name, file_name)
    with open(f"data/{path}", "rb") as input_file:
        one_len_list = pickle.load(input_file)
    return one_len_list


def read_all_from_dir(folder):
    full_sequence_in_folder = []
    check_files_path = os.path.join('data/', folder)
    files = os.listdir(check_files_path)
    for file in files:
        full_one_len_list = clay_pkl_files(folder, file)
        full_sequence_in_folder.append(full_one_len_list)
    return full_sequence_in_folder


if __name__ == "__main__":
    before_60 = read_all_from_dir('pkl_from_parser')
    print('len(before_60):', len(before_60))
    more_60 = read_all_from_dir('pkl_from_parser_prl')
    print('len(more_60):', len(more_60))
    more_40_list = before_60 + more_60
    print('len(all):', len(more_40_list))
    with open(f"data/seq_40_more.pkl", 'wb') as f:
        pickle.dump(more_40_list, f)
    print('ok')
