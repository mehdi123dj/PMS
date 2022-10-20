from utils import getText, clean_text, segmente, store_data
import argparse
from os.path import isfile, join
from os import listdir
import pandas as pd 
import spacy
import os 
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from utils import functionize
from sklearn.model_selection import train_test_split


def main():
    """
    Collect arguments and run.
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-dp",
        "--data-path",
        help="The path to the paragraphs data set",
        default="./data",
    )

    args = parser.parse_args()
    root = os.path.join(args.data_path,"Fichiers_textes_codés_pour_Datarvest")
    files = [join(root,f) for f in listdir(root) if isfile(join(root, f))]

    code_dict = {}
    for f in list(set(files)-{join(args.data_path,'Codes_164_auteurs_Lidia_IA.xlsx') }):
        l = segmente(getText(f))
        for t,c in l:
            code_dict[t] = c

    store_data(code_dict, args.data_path)

    nlp = spacy.load('fr_core_news_md')
    file_path = os.path.join(args.data_path,"paragraphs.csv")
    df = pd.read_csv(file_path,names  = ['text','noyau','code'])

    df_lemmatized  = df.copy()
    for i in range(len(df)):
        t = df.iloc[i]['text']
        tokens = [token.lemma_.lower() for token in nlp(t) if token.lemma_ in fr_stop]
        t_new = ''
        for token in tokens: 
            t_new += token + ' '
        df_lemmatized = df_lemmatized.replace(df_lemmatized.iloc[i]['text'],clean_text(t_new[:-1]))

    df_lemmatized.to_csv(args.data_path + '/lem_paragraphs_stop.csv',sep=',', index=False)

    df_functions = pd.read_excel(os.path.join(args.data_path,"PMS_Dictionnaire_fonctions_pour_Datarvest_131221_ok.xlsx"),header=None)
    df_functions = df_functions.iloc[df_functions[1].dropna().index]
    df_functions = df_functions.reset_index(drop=True)
    df_functions[1] = [df_functions[1].iloc[i].lower() for i in range(len(df_functions))]

    functions = df_functions.iloc[df_functions.iloc[df_functions[1].dropna().index][3].dropna().index][[1,3]]
    functions = functions[functions[3].isin(['S','G','P'])]
    functions = functions[functions[1].isin(fr_stop)]
    functions = functions.reset_index(drop=True)
    for i in range(len(functions)):
        functions.iloc[i][1] = [token.lemma_ for token in nlp(functions.iloc[i][1])][0]
    functions_dict = {functions.iloc[i][1]: functions.iloc[i][3] for i in range(len(functions))}

    data_df = df_lemmatized
    data_df['G'] = [functionize(data_df.iloc[i]['text'],functions_dict,'G') for i in range(len(data_df))]
    data_df['S'] = [functionize(data_df.iloc[i]['text'],functions_dict,'S') for i in range(len(data_df))]
    data_df['P'] = [functionize(data_df.iloc[i]['text'],functions_dict,'P') for i in range(len(data_df))]
    data_df['NA'] = [functionize(data_df.iloc[i]['text'],functions_dict,'NA') for i in range(len(data_df))]

    index_to_cut = []
    for i in range(len(data_df)): 
        if min([len(data_df.iloc[i].G),len(data_df.iloc[i].S),len(data_df.iloc[i].P)])==0:
            index_to_cut.append(i)

    data_df = data_df.iloc[[i for i in range(len(data_df)) if i not in index_to_cut]]
    data_df = data_df.reset_index(drop=True)

    train_test_ratio = 0.90
    train_valid_ratio = 8/9

    df_full_train, df_test = train_test_split(data_df, train_size = train_test_ratio, random_state = 1)
    df_train, df_valid = train_test_split(df_full_train, train_size = train_valid_ratio, random_state = 1)

    df_full = data_df[["noyau","text","G","S","P","NA"]]
    df_train = df_train[["noyau","text","G","S","P","NA"]]
    df_valid = df_valid[["noyau","text","G","S","P","NA"]]
    df_test = df_test[["noyau","text","G","S","P","NA"]]

    df_train.to_csv(args.data_path + '/train_lem_stop.csv',sep=',', index=False)
    df_valid.to_csv(args.data_path + '/valid_lem_stop.csv',sep=',', index=False)
    df_test.to_csv(args.data_path + '/test_lem_stop.csv',sep=',', index=False)


if __name__ == "__main__":
    main()
