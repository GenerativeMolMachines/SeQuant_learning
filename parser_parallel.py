import pickle
import requests
import numpy as np
from joblib import Parallel, delayed


base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
search_url = base_url + "esearch.fcgi"
fetch_url = base_url + "efetch.fcgi"
aa_set = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
              'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y'}
# Step 1: Perform the search to get the IDs of matching sequences
all_sequences = []
def parser_by_len(length):
    search_params = {
        "db": "protein",
        "term": f"({length}[SLEN] AND ((animals[filter] OR bacteria[filter]))",
        "retmax": 250000,  # Adjust retmax as needed
        "retmode": "json"
    }
    response = requests.get(search_url, params=search_params)
    search_results = response.json()


    if int(search_results["esearchresult"]["count"]) == 0:
        print(f"No sequences found with the specified length={length}.")
        return

    id_list = search_results["esearchresult"]["idlist"]

    several_id_lists = np.array_split(np.asarray(id_list), 2500)
    seq_list = []
    # Step 2: Fetch the sequences using the IDs
    for id_l in several_id_lists:
        if len(seq_list) > 80000:
            print('more then break')
            break
        fetch_params = {
            "db": "protein",
            "id": ",".join(list(id_l)),
            "rettype": "fasta",
            "retmode": "text"
        }
        response = requests.get(fetch_url, params=fetch_params)

        if response.ok:
            sequences_text = response.text.split('\n\n')[:-1]
            for b in sequences_text:
                c = b.split('\n')
                s = c[1]
                if s not in seq_list and set(s).issubset(aa_set):
                    seq_list.append(c[1])
        else:
            print(f"Failed to fetch sequences length={length}: {response.status_code} - {response.reason}")

    with open(f"data/pkl_from_parser_prl/seq_{length}_{len(seq_list)}.pkl", 'wb') as f:
        pickle.dump(seq_list, f)

    all_sequences.extend(seq_list)

func_out = Parallel(n_jobs=-1)(
    [
        delayed(parser_by_len)(
            length
        ) for length in range(41, 97, 1)
    ]
)

with open(f"seq_all_parser_{len(all_sequences)}_41_96_prl.pkl", 'wb') as f:
    pickle.dump(all_sequences, f)
