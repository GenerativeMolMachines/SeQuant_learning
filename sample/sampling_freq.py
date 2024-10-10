import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Loading balanced dataset
with open('../data/test_seq_clean_str.pkl', 'rb') as f:
    test_data = pickle.load(f)

with open('../data/train_seq_clean_str.pkl', 'rb') as f:
    train_data = pickle.load(f)

print('data has been imported')

test_data = pd.DataFrame(test_data, columns=['sequence'])
train_data = pd.DataFrame(train_data, columns=['sequence'])

# Possible amino acids
amino_acids = sorted('ARNDCEQGHILKMFPSTWYVOU')
scaler = MinMaxScaler(feature_range=(0, 1))

random_state = 42


def encode_sequences_with_frequency(df, sequence_column='sequence'):
    def compute_frequency_vector(sequence):
        vector = np.zeros(len(amino_acids), dtype=float)
        for amino_acid in sequence:
            if amino_acid in amino_acids:
                index = amino_acids.index(amino_acid)
                vector[index] += 1
        vector /= len(sequence)
        return vector

    frequency_vectors = df[sequence_column].apply(compute_frequency_vector).tolist()

    scaled_vectors = scaler.fit_transform(frequency_vectors)

    return np.array(scaled_vectors)


def clustering(df, sequence_column='sequence', n_clusters=22):
    encoded_sequences = encode_sequences_with_frequency(df, sequence_column)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['cluster'] = kmeans.fit_predict(encoded_sequences)

    # AA frequency in full dataset
    full_sequence = ''.join(df[sequence_column])
    full_amino_acid_counts = {aa: full_sequence.count(aa) for aa in amino_acids}
    total_full_count = len(full_sequence)

    cluster_labels = {}
    for cluster in range(n_clusters):
        cluster_sequences = df[df['cluster'] == cluster][sequence_column]
        cluster_sequence_str = ''.join(cluster_sequences)

        # Each aa's frequency in current cluster
        cluster_amino_acid_counts = {aa: cluster_sequence_str.count(aa) for aa in amino_acids}
        total_cluster_count = len(cluster_sequence_str)

        # AA with max ratio
        max_ratio = -1
        best_amino_acid = None
        for aa in sorted(amino_acids):
            if full_amino_acid_counts[aa] > 0:
                ratio = (cluster_amino_acid_counts[aa] / total_cluster_count) / (
                            full_amino_acid_counts[aa] / total_full_count)
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_amino_acid = aa

        cluster_labels[cluster] = best_amino_acid

    df['cluster_label'] = df['cluster'].map(cluster_labels)

    # Unite clusters with same label
    unique_labels = sorted(set(cluster_labels.values()))
    label_to_combined_cluster = {label: idx for idx, label in enumerate(unique_labels)}
    df['combined_cluster'] = df['cluster_label'].map(label_to_combined_cluster)

    df = df.drop(columns=['cluster'])
    final_num_clusters = len(unique_labels)

    return df, final_num_clusters


def stratified_sampling_with_clustering(
        df, sequence_column='sequence', small_fraction=0.01, medium_fraction=0.1, n_clusters=22, random_state=42
):
    # Clustering
    df, final_num_clusters = clustering(df, sequence_column=sequence_column, n_clusters=n_clusters)

    # Define unique groups by cluster
    unique_clusters = df['combined_cluster'].unique()

    small_sampled_dataframes = []
    medium_sampled_dataframes = []
    large_sampled_dataframes = []

    # Determine min cluster size
    min_cluster_size = min(df[df['combined_cluster'] == cluster].shape[0] for cluster in unique_clusters)

    # Sampling
    for cluster in unique_clusters:
        cluster_df = df[df['combined_cluster'] == cluster].copy()

        small_sample_size = int(min_cluster_size * small_fraction)
        medium_sample_size = int(min_cluster_size * medium_fraction)

        if small_sample_size > 0:
            small_sampled_cluster_df = cluster_df.sample(n=small_sample_size, random_state=random_state)
            small_sampled_dataframes.append(small_sampled_cluster_df)
        else:
            small_sampled_cluster_df = pd.DataFrame()

        if medium_sample_size > 0:
            remaining_after_small = cluster_df.drop(small_sampled_cluster_df.index)
            medium_sampled_cluster_df = remaining_after_small.sample(n=medium_sample_size, random_state=random_state)
            medium_sampled_dataframes.append(medium_sampled_cluster_df)
        else:
            medium_sampled_cluster_df = pd.DataFrame()

        # Remaining data for large sample
        remaining_after_medium = cluster_df.drop(small_sampled_cluster_df.index).drop(medium_sampled_cluster_df.index)
        if min_cluster_size > 0:
            large_sampled_cluster_df = remaining_after_medium.sample(
                n=int(min_cluster_size * (1 - small_fraction - medium_fraction)), random_state=random_state)
            large_sampled_dataframes.append(large_sampled_cluster_df)

    print(f"Unique clusters: {unique_clusters}")
    print(f"Min cluster size: {min_cluster_size}")
    print(f"Small sample size: {small_sample_size}, Medium sample size: {medium_sample_size}")

    # Merge all samples
    small_sampled_df = pd.concat(small_sampled_dataframes).reset_index(drop=True)
    medium_sampled_df = pd.concat(medium_sampled_dataframes).reset_index(drop=True)
    large_sampled_df = pd.concat(large_sampled_dataframes).reset_index(drop=True)

    print(f"Small sampled dataframes count: {len(small_sampled_df)}")
    print(f"Medium sampled dataframes count: {len(medium_sampled_df)}")
    print(f"Large sampled dataframes count: {len(large_sampled_df)}")

    return small_sampled_df, medium_sampled_df, large_sampled_df
    

# Functions implementation
small_train_df, medium_train_df, large_train_df = stratified_sampling_with_clustering(
    train_data, sequence_column='sequence', small_fraction=0.1, medium_fraction=0.3, n_clusters=22
)

print('train set has been splitted')

small_train_df.to_csv('../data/small_train_df.csv', index=False)
medium_train_df.to_csv('../data/medium_train_df.csv', index=False)
large_train_df.to_csv('../data/large_train_df.csv', index=False)

print('train data has been saved')

small_test_df, medium_test_df, large_test_df = stratified_sampling_with_clustering(
    test_data, sequence_column='sequence', small_fraction=0.1, medium_fraction=0.3, n_clusters=22
)

print('test set has been splitted')

small_test_df.to_csv('../data/small_test_df.csv', index=False)
medium_test_df.to_csv('../data/medium_test_df.csv', index=False)
large_test_df.to_csv('../data/large_test_df.csv', index=False)

print('test data has been saved')
