import pickle
import os


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


if __name__ == "__main__":
    select_folder()
