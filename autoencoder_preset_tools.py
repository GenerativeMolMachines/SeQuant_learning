import pandas as pd
import numpy as np
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.preprocessing import MinMaxScaler

monomer_dict = {
    'A': 'CC(N)C(=O)O', 'R': 'NC(N)=NCCCC(N)C(=O)O', 'N': 'NC(=O)CC(N)C(=O)O',
    'D': 'NC(CC(=O)O)C(=O)O', 'C': 'NC(CS)C(=O)O', 'Q': 'NC(=O)CCC(N)C(=O)O',
    'E': 'NC(CCC(=O)O)C(=O)O', 'G': 'NCC(=O)O', 'H': 'NC(Cc1cnc[nH]1)C(=O)O',
    'I': 'CCC(C)C(N)C(=O)O', 'L': 'CC(C)CC(N)C(=O)O', 'K': 'NCCCCC(N)C(=O)O',
    'M': 'CSCCC(N)C(=O)O', 'F': 'NC(Cc1ccccc1)C(=O)O', 'P': 'O=C(O)C1CCCN1',
    'S': 'NC(CO)C(=O)O', 'T': 'CC(O)C(N)C(=O)O', 'W': 'NC(Cc1c[nH]c2ccccc12)C(=O)O',
    'Y': 'NC(Cc1ccc(O)cc1)C(=O)O', 'V': 'CC(C)C(N)C(=O)O', 'O': 'CC1CC=NC1C(=O)NCCCCC(N)C(=O)O',
    'U': 'NC(C[Se])C(=O)O'
}

def make_monomer_descriptors() -> pd.DataFrame:
    descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
    get_descriptors = rdMolDescriptors.Properties(descriptor_names)
    num_descriptors = len(descriptor_names)

    descriptors_set = np.empty((0, num_descriptors), float)

    for _, value in monomer_dict.items():
        molecule = Chem.MolFromSmiles(value)
        descriptors = np.array(get_descriptors.ComputeProperties(molecule)).reshape((-1,num_descriptors))
        descriptors_set = np.append(descriptors_set, descriptors, axis=0)

    sc = MinMaxScaler(feature_range=(0, 1))
    scaled_array = sc.fit_transform(descriptors_set)
    descriptors_set = pd.DataFrame(scaled_array, columns=descriptor_names, index=monomer_dict.keys())

    energy_data = pd.read_csv('energy_data.csv')
    energy_set = energy_data.set_index("Aminoacid").iloc[:, :]

    energy_names = energy_set.columns

    scaled_energy = sc.fit_transform(energy_set)
    scaled_energy_set = pd.DataFrame(scaled_energy, columns=energy_names, index=monomer_dict.keys())

    all_descriptors = pd.concat([descriptors_set, scaled_energy_set], axis=1)
    return all_descriptors


def seq_to_matrix(
    sequence: str,
    descriptors: pd.DataFrame,
    num: int
):
    rows = descriptors.shape[1]
    seq_matrix = np.empty((0, rows), float)  # shape (0,rows)
    for aa in sequence:
        if len(aa) > 1:
            print(aa)
            print(sequence)
        descriptors_array = np.array(descriptors.loc[aa]).reshape((1, rows))  # shape (1,rows)
        seq_matrix = np.append(seq_matrix, descriptors_array, axis=0)
    seq_matrix = seq_matrix.T
    shape = seq_matrix.shape[1]
    if shape < num:
        add_matrix = np.pad(seq_matrix,
                            [(0, 0), (0, num-shape)],
                            mode='constant',
                            constant_values=0)
        #water = np.array(descriptors.loc['water']).reshape((rows,1))
        #water_padding = np.resize(a=water, new_shape=(rows,num-shape))
        #add_matrix = np.concatenate((seq_matrix,water_padding), axis=1)

        return add_matrix  # shape (rows,n)

    return seq_matrix


def encode_seqs(
    sequences_list: list[str],
    descriptors: pd.DataFrame,
    num: int
):
    lst = []
    i = 0
    for sequence in sequences_list:
        seq_matrix = seq_to_matrix(sequence=sequence, descriptors=descriptors, num=num)
        lst.append(seq_matrix)
        i += 1
    encoded_seqs = np.dstack(lst)

    return encoded_seqs


def preprocess_input(peptides):
    peptides = peptides.reshape((peptides.shape[0], peptides.shape[1], peptides.shape[2], 1))
    return peptides


def train_test_split(peptides, train_data_ratio):
    indices = peptides.shape[0]
    n_samples = int(indices * train_data_ratio)
    indices = list(range(indices))
    idx_train = np.random.choice(indices, n_samples, replace=False)
    idx_test = list(set(indices) - set(idx_train))
    train_data = peptides[idx_train]
    test_data = peptides[idx_test]
    return train_data, test_data


def filter_sequences(
        sequences: np.array,
        known_symbols: dict[str, str]
):
    filtered_sequences = []
    for seq in sequences:
        if set(seq).issubset(set(known_symbols)):
            filtered_sequences.append(seq)
    return filtered_sequences


def data_processing(
        batch_data: list[str],
        max_len: int
):
    descriptors_set = make_monomer_descriptors()
    batch_encoded_sequences = encode_seqs(batch_data, descriptors_set, max_len)
    batch_encoded_sequences = np.moveaxis(batch_encoded_sequences, -1, 0)

    batch_processed = preprocess_input(batch_encoded_sequences)

    return batch_processed


def batch_creation_arch(
        data: list[str],
        batch_size: int
) -> list[list[str]]:
    """
    Coverts initial data (list of sequences) to the batches
    :param data: list of polymer sequences
    :param batch_size: number of sequences in a batch
    :return: list of lists with batch_size sequences in each
    """
    num_batches = len(data) // batch_size
    lengths = [len(seq) for seq in data]

    data_by_length = {}
    for length, seq in zip(lengths, data):
        if length not in data_by_length:
            data_by_length[length] = []
        if len(seq) == length:
            data_by_length[length].append(seq)
        else:
            print(f'Length of the sequence {seq} does not match with key length {length}')

    batches = [[] for _ in range(num_batches)]

    for length, seqs in data_by_length.items():
        np.random.shuffle(seqs)

        for length, seqs in data_by_length.items():
            np.random.shuffle(seqs)

            for i, seq in enumerate(seqs):
                batches[i % num_batches].append(seq)

        return batches


def create_dataset_from_batches(
        batches: list[list[str]],
        monomer_dict: dict[str, str],
        max_len: int
):
    def generator():
        for batch in batches:
            processed_batch = data_processing(batch, monomer_dict, max_len)
            yield processed_batch, processed_batch

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),  # input
            tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)  # target
        )
    )
    return dataset


def oversampling(
        sequences: list[str],
        target_divisor: int
):
    current_size = len(sequences)
    remainder = current_size % target_divisor

    if remainder == 0:
        print("Dataset size is already equal to ", target_divisor)
        return sequences

    additional_records_needed = target_divisor - remainder

    sampled_indices = np.random.choice(len(sequences), size=additional_records_needed, replace=True)
    oversampled_sequences = [sequences[i] for i in sampled_indices]
    print(oversampled_sequences)
    result_sequences = sequences + oversampled_sequences

    return result_sequences
