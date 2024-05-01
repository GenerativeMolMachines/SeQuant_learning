import os
import json

len_dict = {}
for folder in ['pkl_from_parser', 'pkl_from_parser_prl']:
    check_files_path = os.path.join('data/', folder)
    files = os.listdir(check_files_path)
    for file_name in files:
        name_split = file_name.split('_')
        real_len = name_split[2].split('.')
        len_dict[name_split[1]] = int(real_len[0])

with open('len_dict_40_96.json', 'w') as fp:
    json.dump(len_dict, fp)

