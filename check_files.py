import pickle

with open(r"data/seq_40_96.pkl", "rb") as input_file:
    already_exist = pickle.load(input_file)
print(len(already_exist))
