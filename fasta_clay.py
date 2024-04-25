from Bio import SeqIO
import pickle
# from joblib import Parallel, delayed


def make_pickle_files(
    sequence_files_name: list,
    ready_list: list,
    known_aa: set
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


if __name__ == "__main__":
    with open(r"data/seq_40_96.pkl", "rb") as input_file:
        already_exist = pickle.load(input_file)
    aa_set = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
              'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y'}

    clean_ready = [seq for seq in already_exist if set(seq).issubset(aa_set)]
    print('clean_ready', len(clean_ready))
    files_names_40_96 = [f"data/sequence_{num}.fasta" for num in range(40, 97, 1)]
    make_pickle_files(files_names_40_96, clean_ready, aa_set)
# func_out = Parallel(n_jobs=-1)(
#     [
#         delayed(make_pickle_files)(
#             file_name
#         ) for file_name in files_names_40_96
#     ]
# )
#     print(len(list_seq))
#     with open("seq_40_96.pkl", 'wb') as f:
#         pickle.dump(list_seq, f)
