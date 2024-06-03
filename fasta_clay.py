from Bio import SeqIO
import pickle
import gzip
# from joblib import Parallel, delayed

aa_set = {'A', 'C', 'G', 'T'}

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

    # with open("rna_11.pkl", 'wb') as f:
    #     pickle.dump(sequences_11, f)
    # with open("rna_12.pkl", 'wb') as f:
    #     pickle.dump(sequences_12, f)
    #
    # with open("rna_13.pkl", 'wb') as f:
    #     pickle.dump(sequences_13, f)

    return



if __name__ == "__main__":
    # with open(r"data/seq_40_96.pkl", "rb") as input_file:
    #     already_exist = pickle.load(input_file)
    # aa_set = {'A', 'C', 'G', 'T'}
    #
    # clean_ready = [seq for seq in already_exist if set(seq).issubset(aa_set)]
    # print('clean_ready', len(clean_ready))
    # files_names_40_96 = [f"data/sequence_{num}.fasta" for num in range(40, 97, 1)]
    # make_pickle_files(files_names_40_96, clean_ready, aa_set)
    read_fasta_file('data/sequence.fasta')
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
