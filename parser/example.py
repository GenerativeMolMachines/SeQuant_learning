from joblib import Parallel, delayed
from prot_parser import parser_by_len
from file_tools import select_folder


# Parser by len from list
func_out = Parallel(n_jobs=2)(
    [
        delayed(parser_by_len)(
            length
        ) for length in [88, 90]
    ]
)

# check sequences in folders
select_folder()