from utils import getText, clean_text, segmente, store_data
import argparse
from os.path import isfile, join
from os import listdir
import pandas as pd 



def main():
    """
    Collect arguments and run.
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-dp",
        "--data-path",
        help="The path to the paragraphs data set",
        default="../data/Fichiers_textes_codés_pour_Datarvest",
    )
    parser.add_argument(
        "-sp",
        "--store-path",
        help="The path in which the formated data set is stored",
        default="../data",
    )
    args = parser.parse_args()

    root = args.data_path
    files = [join(root,f) for f in listdir(root) if isfile(join(root, f))]

    code_dict = {}
    for f in list(set(files)-{join(args.data_path,'Codes_164_auteurs_Lidia_IA.xlsx') }):
        l = segmente(getText(f))

        for t,c in l:
            code_dict[clean_text(t)] = c
            
            
    store_data(code_dict, args.store_path)

if __name__ == "__main__":
    main()