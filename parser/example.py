from joblib import Parallel, delayed
from prot_parser import parser_by_len
from file_tools import select_folder


# Parser by len from list
# func_out = Parallel(n_jobs=2)(
#     [
#         delayed(parser_by_len)(
#             length
#         ) for length in [5, 6]
#     ]
# )
len_list = [5, 6, 22, 23, 24, 25, 26, 12, 17, 18, 19, 20, 21, 38]
for length in len_list:
    parser_by_len(length=length)

# check sequences in folders
# select_folder()
