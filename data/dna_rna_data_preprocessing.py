import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Variables
np.random.seed(2024)
n_samples = 1200
start_length = 14
end_length = 96

# Import data
dna_df = pd.read_csv('dna_rna/sequences DNA.txt', header=None)
rna_df = pd.read_csv('dna_rna/sequences RNA.txt', header=None)

dna_df.columns = ['sequence']
rna_df.columns = ['sequence']

protein = pd.read_csv('dna_rna/prot_14_to_96.csv')
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
