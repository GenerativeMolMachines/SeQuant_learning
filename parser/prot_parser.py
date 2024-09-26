import pickle
import random
import requests
import numpy as np
import time


base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
search_url = base_url + "esearch.fcgi"
fetch_url = base_url + "efetch.fcgi"
aa_set = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
              'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y'}
proxies_list = [
        '111.111.111.111:2222',
        '333.333.333.333:4444',
        '444.444.444.444:5555',
        '777.777.777.777:8888',
        '8888.8888.8888.8888:777',
        '444.333.444.333:5555',
        '777.5555.5555.777:8888',
        '919.919.919.919:0000'
    ]

with open('test_seq_clean_str.pkl', 'rb') as f:
    test_data = pickle.load(f)

print("test_data_len = " + str(len(test_data)))


# Step 1: Perform the search to get the IDs of matching sequences
def parser_by_len(
    length: int,
    retmax: int = 500000,
    max_samples_amount: int = 80000,
    path_to_save: str = 'data/parse_bad_len_sep'
):
    """
    Parser for protein sequences from NCBI. Saves file in dir permanently.
    :param length: sequence length
    :param retmax: number of unique names to retrieve (does not always mean uniqueness of the sequence)
    :param arrays_amount: number of arrays to divide. Division is necessary because no
    more than 100 names are sent per request. Depends on retmax
    :param max_samples_amount: number of unique sequences in the final set. When this number is reached, parsing is forcibly terminated
    :param path_to_save: storage folder
    :return: None
    """
    search_params = {
        "db": "protein",
        "term": f"{length}[SLEN]",
        "retmax": retmax,  # Adjust retmax as needed
        "retmode": "json"
    }
    proxies = {
        'http': random.choice(proxies_list)
    }
    response = requests.get(search_url, params=search_params, proxies=proxies)
    search_results = response.json()

    if search_results.get('error', '') != '':
        print(search_results.get('error'))
        print(f"len={length} failed")
        return
    else:
        print(f"len={length} ok")

    if int(search_results["esearchresult"]["count"]) == 0:
        print(f"No sequences found with the specified length={length}.")
        return

    id_list = search_results["esearchresult"]["idlist"]
    # num of arrays (9k for here) depends on retmax, num of val in arr must be <100
    several_id_lists = np.array_split(np.asarray(id_list), int(len(id_list) / 50) + 1)

    seq_list = []
    # Step 2: Fetch the sequences using the IDs
    for id_l in several_id_lists:
        if len(seq_list) > max_samples_amount:  # 80k - num which we need, could be changed
            print('more then break')
            break
        fetch_params = {
            "db": "protein",
            "id": ",".join(list(id_l)),
            "rettype": "fasta",
            "retmode": "text"
        }

        try:
            proxies = {
                'http': random.choice(proxies_list)
            }
            time.sleep(1)
            response = requests.get(fetch_url, params=fetch_params, proxies=proxies)
        except:
            continue

        if response.ok:
            sequences_text = response.text.split('\n\n')[:-1]

            for b in sequences_text:
                c = b.split('\n')
                s = c[-1]
                if (s not in seq_list) and (set(s).issubset(aa_set)) and (s not in test_data):
                    seq_list.append(s)
        else:
            print(f"Failed to fetch sequences length={length}: {response.status_code} - {response.reason}")

    file_name = f"/seq_{length}_{len(seq_list)}.pkl"
    path = path_to_save + file_name
    with open(path, 'wb') as f:
        pickle.dump(seq_list, f)
