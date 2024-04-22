import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

aa_dict = {'A': 'CC(C(=O)O)N', 'R': 'C(CC(C(=O)O)N)CN=C(N)N', 'N': 'C(C(C(=O)O)N)C(=O)N',
           'D': 'C(C(C(=O)O)N)C(=O)O', 'C': 'C(C(C(=O)O)N)S', 'Q': 'C(CC(=O)N)C(C(=O)O)N',
           'E': 'C(CC(=O)O)C(C(=O)O)N', 'G': 'C(C(=O)O)N', 'H': 'C1=C(NC=N1)CC(C(=O)O)N',
           'I': 'CCC(C)C(C(=O)O)N', 'L': 'CC(C)CC(C(=O)O)N', 'K': 'C(CCN)CC(C(=O)O)N',
           'M': 'CSCCC(C(=O)O)N', 'F': 'C1=CC=C(C=C1)CC(C(=O)O)N', 'P': 'C1CC(NC1)C(=O)O',
           'S': 'C(C(C(=O)O)N)O', 'T': 'CC(C(C(=O)O)N)O', 'W': 'C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N',
           'Y': 'C1=CC(=CC=C1CC(C(=O)O)N)O', 'V': 'CC(C)C(C(=O)O)N', 'O': 'CC1CC=NC1C(=O)NCCCCC(C(=O)O)N',
           'U': 'C(C(C(=O)O)N)[Se]'}
n = 96

# Processing labeled data (AMPs database)
labeled_data = pd.read_csv('data/AMP_ADAM2.txt', on_bad_lines='skip')
labeled_data = labeled_data.replace('+', 1)
labeled_data = labeled_data.fillna(0)
labeled_data = labeled_data.drop(labeled_data[labeled_data.SEQ.str.contains(r'[@#&$%+-/*BXZ]')].index)
labeled_data_seqs = labeled_data['SEQ'].to_list()

# # Processing unlabeled data (Non-AMPs and CPPBase)
# with open('../data/Non-AMPs.txt') as f:
#     file = f.readlines()
# raw_seqs = file[1::2]
# unlabeled_data = [s.replace("\n", "") for s in raw_seqs]
#
# with open('../data/natural_pep (cpp).txt') as f:
#     file = f.readlines()
# raw_seqs = file[1::2]
# unlabeled_data_2 = [s.replace("\n", "") for s in raw_seqs]
#
# with open('../data/uniprot_sprot.txt') as f:
#     raw_seqs_2 = f.readlines()
# unlabeled_data_3 = [s.replace("\n", "") for s in raw_seqs_2]
#
# with open('../data/SPENCER_ORF_protein_sequence.txt') as f:
#     raw_seqs_2 = f.readlines()
# unlabeled_data_4 = [s.replace("\n", "") for s in raw_seqs_2]
#
# with open('../data/hspvfullR58HET.txt') as f:
#     raw_seqs_2 = f.readlines()
# unlabeled_data_5 = [s.replace("\n", "") for s in raw_seqs_2]

# DBAASP database
# dbaasp = pd.read_csv('../data/peptides.csv')
# dbaasp_2 = pd.read_csv('../data/peptides_2.csv')
# unlabeled_data_6 = list(
#     dict.fromkeys(dbaasp['SEQUENCE'].astype('str').tolist() + dbaasp_2['SEQUENCE'].astype('str').tolist()))


# Merged data for CAE training
# all_seqs = labeled_data_seqs + unlabeled_data + unlabeled_data_2 + unlabeled_data_3 + unlabeled_data_4 + unlabeled_data_5 + unlabeled_data_6
all_seqs = labeled_data_seqs
all_seqs = [seq for seq in all_seqs if len(seq) <= n]
all_seqs = list(dict.fromkeys(all_seqs))
all_seqs = [string.upper() for string in all_seqs]
all_seqs_full = [x for x in all_seqs if "B" not in x
            and "X" not in x
            and "Z" not in x
            and "5" not in x
            and "8" not in x
            and "-" not in x
            and " " not in x]
df = pd.DataFrame()
df['s'] = all_seqs_full
del all_seqs_full
del labeled_data_seqs
# del unlabeled_data
# del unlabeled_data_2
# del unlabeled_data_3
# del unlabeled_data_4
# del unlabeled_data_5
# del unlabeled_data_6

df['len']  = df['s'].str.len()
for i in aa_dict.keys():
  df[i] = df['s'].str.count(i)

rand_st = 0
train, test = train_test_split(df, shuffle=True, test_size=0.3, random_state=rand_st)

for i in range(150):
        train, test = train_test_split(df, shuffle=True, test_size=0.3, random_state=rand_st)
        train_distr = train.describe()
        train_distr_mean = train_distr.loc['mean']

        test_distr = test.describe()
        test_distr_mean = test_distr.loc['mean']

        result_minus = abs(test_distr_mean-train_distr_mean)
        aa_distribution = result_minus.drop('len')
        if test_distr.loc['max', 'len'] > train_distr.loc['max', 'len']:
            rand_st += 1
            continue
        elif aa_distribution[aa_distribution>0.1].shape[0] > 0:
            rand_st += 1
            continue
        elif result_minus['len'] > 0.5:
            rand_st += 1
            continue

        else:
            break
test_distr.to_csv('test_distr.csv')
train_distr.to_csv('train_distr.csv')

train_seq_ = train['s'].to_list()
with open('train_seq_.pkl', 'wb') as f:
   pickle.dump(train_seq_, f)
del train_seq_

test_seq_ = test['s'].to_list()
with open('test_seq_.pkl', 'wb') as f:
   pickle.dump(test_seq_, f)