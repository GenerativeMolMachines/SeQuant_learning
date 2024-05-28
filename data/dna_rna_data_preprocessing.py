import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from autoencoder_preset_tools import filter_sequences

# Variables
np.random.seed(2024)
n_samples = 1200
start_length = 14
end_length = 96

dna_dict = {
    'A': r'O=P(O)(O)OP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n2cnc1c(ncnc12)N)C[C@@H]3O',  # DNA
    'T': r'CC1=CN(C(=O)NC1=O)C2CC(C(O2)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O',
    'G': r'O=P(O)(O)OP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n1cnc2c1NC(=N/C2=O)\N)C[C@@H]3O',
    'C': r'C1[C@@H]([C@H](O[C@H]1N2C=CC(=NC2=O)N)CO[P@@](=O)(O)O[P@@](=O)(O)OP(=O)(O)O)O',
}

rna_dict = {
    'A': r'c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N',  # RNA
    'U': r'C1=CN(C(=O)NC1=O)C2C(C(C(O2)COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])O)O',
    'G': r'C1=NC2=C(N1C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N=C(NC2=O)N',
    'C': r'c1cn(c(=O)nc1N)[C@H]2[C@@H]([C@@H]([C@H](O2)CO[P@](=O)(O)O[P@](=O)(O)OP(=O)(O)O)O)O'
}

protein_dict = {
    'A': 'CC(C(=O)O)N',  # protein
    'R': 'C(CC(C(=O)O)N)CN=C(N)N', 'N': 'C(C(C(=O)O)N)C(=O)N', 'D': 'C(C(C(=O)O)N)C(=O)O', 'C': 'C(C(C(=O)O)N)S',
    'Q': 'C(CC(=O)N)C(C(=O)O)N', 'E': 'C(CC(=O)O)C(C(=O)O)N', 'G': 'C(C(=O)O)N',
    'H': 'C1=C(NC=N1)CC(C(=O)O)N', 'I': 'CCC(C)C(C(=O)O)N', 'L': 'CC(C)CC(C(=O)O)N',
    'K': 'C(CCN)CC(C(=O)O)N', 'M': 'CSCCC(C(=O)O)N', 'F': 'C1=CC=C(C=C1)CC(C(=O)O)N', 'P': 'C1CC(NC1)C(=O)O',
    'S': 'C(C(C(=O)O)N)O', 'T': 'CC(C(C(=O)O)N)O', 'W': 'C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N',
    'Y': 'C1=CC(=CC=C1CC(C(=O)O)N)O', 'V': 'CC(C)C(C(=O)O)N', 'O': 'CC1CC=NC1C(=O)NCCCCC(C(=O)O)N',
    'U': 'C(C(C(=O)O)N)[Se]'
}

# Import data
dna_df = pd.read_csv('dna_rna/sequences DNA.txt', header=None)
rna_df = pd.read_csv('dna_rna/sequences RNA.txt', header=None)
protein = pd.read_csv('dna_rna/prot_14_to_96.csv')

# Filtering sequences
dna_df = filter_sequences(sequences=dna_df[0].tolist(), known_symbols=dna_dict)
rna_df = filter_sequences(sequences=rna_df[0].tolist(), known_symbols=dna_dict)
protein = filter_sequences(sequences=protein['seq'].tolist(), known_symbols=protein_dict)

dna_df = pd.DataFrame(dna_df)
rna_df = pd.DataFrame(rna_df)
protein = pd.DataFrame(protein)

dna_df.columns = ['sequence']
rna_df.columns = ['sequence']
protein.columns = ['sequence']


# Functions
def sequences_parsing(
        start_length: int,
        end_length: int,
        polymer_df: pd.DataFrame
):
    """
    Function for creating balanced dataframe from the original polymer df with polymer sequences.
    :param start_length: initial polymer length value for parsing
    :param end_length: final polymer length value for parsing
    :param polymer_df: pd.DataFrame which contains one column of polymer sequences
    :return: pd.DataFrame of all parsed sequences stratified by sequences length
    """

    polymer_list = {}
    polymer_missing_lengths = []

    for length in range(start_length, end_length+1):
        try:
            sequences = polymer_df[polymer_df['sequence'].apply(len) == length].drop_duplicates(
                subset='sequence').sample(n=n_samples)
            polymer_list[length] = sequences
        except ValueError:
            polymer_missing_lengths.append(length)

    print(polymer_missing_lengths)
    polymer = pd.concat(polymer_list.values())
    polymer.reset_index(drop=True, inplace=True)

    return polymer


def split(
        polymer_df: pd.DataFrame
):
    """
    Function for train_test_split of stratified by sequences DataFrame
    :param polymer_df: pd.DataFrame with parsed polymer sequences stratified by sequences length
    :return: train and test pd.DataFrames, with test_size = 0.3
    """

    polymer_grouped = polymer_df.groupby(polymer_df['sequence'].apply(len))
    polymer_train_list = []
    polymer_test_list = []

    for length, group in polymer_grouped:
        train_data, test_data = train_test_split(group, test_size=0.3, random_state=2024)
        polymer_train_list.append(train_data)
        polymer_test_list.append(test_data)

    polymer_train = pd.concat(polymer_train_list).reset_index(drop=True)
    polymer_test = pd.concat(polymer_test_list).reset_index(drop=True)

    return polymer_train, polymer_test


# Parsing sequences for each length from 14 to 96
dna = sequences_parsing(start_length=start_length, end_length=end_length, polymer_df=dna_df)
rna = sequences_parsing(start_length=start_length, end_length=end_length, polymer_df=rna_df)

# Forming train/test datasets
dna_train, dna_test = split(polymer_df=dna)
rna_train, rna_test = split(polymer_df=rna)
protein_train, protein_test = split(polymer_df=protein)

dna_train.to_csv('dna_rna/dna_train.csv', index=False)
dna_test.to_csv('dna_rna/dna_test.csv', index=False)

rna_train.to_csv('dna_rna/rna_train.csv', index=False)
rna_test.to_csv('dna_rna/rna_test.csv', index=False)

protein_train.to_csv('dna_rna/protein_train.csv', index=False)
protein_test.to_csv('dna_rna/protein_test.csv', index=False)
