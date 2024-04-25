from Bio import SeqIO
import pickle
from joblib import Parallel, delayed


list_seq = []


def make_pickle_files(
    sequence_files_name: str
):
    one_len_seq_list = []
    fasta_sequences = SeqIO.parse(open(sequence_files_name), 'fasta')

    for fasta in fasta_sequences:
        sequence = str(fasta.seq)
        one_len_seq_list.append(sequence)

    one_len_seq_list_unique = list(set(one_len_seq_list))
    print(sequence_files_name, len(one_len_seq_list_unique))
    list_seq.extend(one_len_seq_list_unique)


files_names_40_96 = [f"data/sequence_{num}.fasta" for num in range(40, 97, 1)]
func_out = Parallel(n_jobs=-1)(
    [
        delayed(make_pickle_files)(
            file_name
        ) for file_name in files_names_40_96
    ]
)
print(len(list_seq))
with open("seq_40_96.pkl", 'wb') as f:
    pickle.dump(list_seq, f)
