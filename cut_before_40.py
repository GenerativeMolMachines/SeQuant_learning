import pickle
import json


aa_set = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
              'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y'}

len_dict = {}
for i in range(1, 40, 1):
    len_dict[str(i)] = 0

with open(r"test_seq_.pkl", "rb") as input_file:
    test_file = pickle.load(input_file)
print('len(test_file):', len(test_file))
with open(r"train_seq_.pkl", "rb") as input_file:
    train_file = pickle.load(input_file)
print('len(train_file):', len(train_file))
before_40_list = []

for seq_file in [test_file, train_file]:
    for seq in seq_file:
        s_ln = len(seq)
        if s_ln < 40 and set(seq).issubset(aa_set):
            len_dict[str(s_ln)] += 1
            before_40_list.append(seq)

print('len(before_40_list):', len(before_40_list))
with open(f"data/seq_40_less.pkl", 'wb') as f:
    pickle.dump(before_40_list, f)
with open('len_dict.json', 'w') as fp:
    json.dump(len_dict, fp)
