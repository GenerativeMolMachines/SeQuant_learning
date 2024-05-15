import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

aa_set = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y'}

with open(r"data/seq_96_all.pkl", "rb") as input_file:
    all_seqs_full = pickle.load(input_file)
df = pd.DataFrame()
df['s'] = all_seqs_full

df['len'] = df['s'].str.len()
for i in aa_set:
    df[i] = df['s'].str.count(i)

for rand_st in range(150):
    train, test = train_test_split(df, shuffle=True, test_size=0.3, random_state=rand_st)
    train_distr = train.describe()
    train_distr_mean = train_distr.loc['mean']

    test_distr = test.describe()
    test_distr_mean = test_distr.loc['mean']

    result_minus = abs(test_distr_mean-train_distr_mean)
    aa_distribution = result_minus.drop('len')
    if test_distr.loc['max', 'len'] > train_distr.loc['max', 'len']:
        continue
    elif aa_distribution[aa_distribution > 0.1].shape[0] > 0:
        continue
    elif result_minus['len'] > 0.5:
        continue
    else:
        break
test_distr.to_csv(f"test_distr_{rand_st}.csv")
train_distr.to_csv(f"train_distr_{rand_st}.csv")

train_seq_ = train['s'].to_list()
with open('train_seq_clean.pkl', 'wb') as f:
    pickle.dump(train_seq_, f)
del train_seq_

test_seq_ = test['s'].to_list()
with open('test_seq_clean_.pkl', 'wb') as f:
    pickle.dump(test_seq_, f)
