from Bio import SeqIO
import pickle
from joblib import Parallel, delayed


def make_pickle_files(
    sequence_files_name: list
):
    list_seq = []
    for file_name in sequence_files_name:
        one_len_seq_list = []
        fasta_sequences = SeqIO.parse(open(file_name), 'fasta')

        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            one_len_seq_list.append(sequence)

        one_len_seq_list_unique = list(set(one_len_seq_list))
        list_seq.extend(one_len_seq_list_unique)

        with open(f"seq_40_96.pkl", 'wb') as f:
            pickle.dump(list_seq, f)
        del one_len_seq_list_unique


if __name__ == "__main__":
    files_names_40_96 = [f"data/sequence_{num}.fasta" for num in range(40, 97, 1)]
    func_out = Parallel(n_jobs=-1)(
        [
            delayed(make_pickle_files)(
                file_name
            ) for file_name in files_names_40_96
        ]
    )
