import pandas as pd
import pickle
from Bio import Entrez, SeqIO

monomer_dict = {
    'A': 'CC(C(=O)O)N', 'R': 'C(CC(C(=O)O)N)CN=C(N)N', 'N': 'C(C(C(=O)O)N)C(=O)N',
    'D': 'C(C(C(=O)O)N)C(=O)O', 'C': 'C(C(C(=O)O)N)S', 'Q': 'C(CC(=O)N)C(C(=O)O)N',
    'E': 'C(CC(=O)O)C(C(=O)O)N', 'G': 'C(C(=O)O)N', 'H': 'C1=C(NC=N1)CC(C(=O)O)N',
    'I': 'CCC(C)C(C(=O)O)N', 'L': 'CC(C)CC(C(=O)O)N', 'K': 'C(CCN)CC(C(=O)O)N',
    'M': 'CSCCC(C(=O)O)N', 'F': 'C1=CC=C(C=C1)CC(C(=O)O)N', 'P': 'C1CC(NC1)C(=O)O',
    'S': 'C(C(C(=O)O)N)O', 'T': 'CC(C(C(=O)O)N)O', 'W': 'C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N',
    'Y': 'C1=CC(=CC=C1CC(C(=O)O)N)O', 'V': 'CC(C)C(C(=O)O)N', 'O': 'CC1CC=NC1C(=O)NCCCCC(C(=O)O)N',
    'U': 'C(C(C(=O)O)N)[Se]'
}


def taking_id_len_sequence(name):
    Entrez.email = "susanjyaks@gmail.com" # Always tell NCBI who you are
    handle = Entrez.esearch(db="protein", term=name, retmax="1")
    record = Entrez.read(handle)
    id_num = record["IdList"][0]
    handle = Entrez.efetch(db="protein", id=id_num, rettype="fasta", retmode="text")
    res = SeqIO.read(handle, 'fasta')
    seq = str(res.seq)
    lengh = len(seq)
    return id_num, lengh, seq


big_seq_df = pd.read_pickle('data/ncbi_gen_clusters.pkl')

X = pd.read_csv('data/PCLA_proteins.txt', sep="\t", header=0)
a = X[X['length'] < 97]
plc_df = a[a['length'] > 40]
df = pd.concat([plc_df, big_seq_df])

prot_list = df['accession'].unique()

full_s = []
for name in prot_list:
    try:
        _, s_len, sequence = taking_id_len_sequence(name=name)
        if set(sequence).issubset(set(monomer_dict)):
            if sequence not in full_s:
                full_s.append(sequence)
                print(f"{name} success")
    except:
        continue

print('len full_s', len(full_s))
with open('clusters_prot_40_90.pkl', 'wb') as f:
    pickle.dump(full_s, f)
