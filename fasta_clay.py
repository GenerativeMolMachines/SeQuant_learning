from Bio import SeqIO
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm


def make_pickle_files(
    sequence_files_name: list,
    min_point,
    max_point
):
    list_seq = []
    for file_name in tqdm(sequence_files_name):
        fasta_sequences = SeqIO.parse(open(file_name), 'fasta')

        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            list_seq.append(sequence)
    with open(f"seq_{min_point}_{max_point}.pkl", 'wb') as f:
        pickle.dump(list_seq, f)


if __name__ == "__main__":
    list_s = []
    start_points = [40, 61, 81]
    end_points = [60, 80, 96]
    files_names_40_60 = [f"data/sequence_{num}.fasta" for num in range(40, 61, 1)]
    files_names_61_80 = [f"data/sequence_{num}.fasta" for num in range(61, 81, 1)]
    files_names_81_96 = [f"data/sequence_{num}.fasta" for num in range(81, 97, 1)]
    all_file_names = [files_names_40_60, files_names_61_80, files_names_81_96]
    func_out = Parallel(n_jobs=-1)(
        [
            delayed(make_pickle_files)(
                file_name, start_point, end_point
            ) for file_name, start_point, end_point in zip(
                all_file_names,
                start_points,
                end_points
            )
        ]
    )
