from utils import getText, clean_text, segmente, store_data
import argparse
from os.path import isfile, join
from os import listdir
import pandas as pd 
import spacy
import os 
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from utils import dictionnize
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
    df = pd.read_csv(file_path,names  = ['text','label','code'])

    df_lemmatized  = df.copy()
    for i in range(len(df)):
        t = df.iloc[i]['text']
        tokens = [token.lemma_.lower() for token in nlp(t) if token.lemma_ not in fr_stop]
        t_new = ''
        for token in tokens: 
            t_new += token + ' '
        df_lemmatized = df_lemmatized.replace(df_lemmatized.iloc[i]['text'],clean_text(t_new[:-1]))

    df_lemmatized.to_csv(args.data_path + '/lem_paragraphs.csv',sep=',', index=False)
    data_df = df_lemmatized
    wiki_file = "donnes_wikidot.csv"
    wikidata_df = pd.read_csv(os.path.join(args.data_path,wiki_file),sep=',')
    wikidot_df = wikidata_df.iloc[[len(wikidata_df['Title'].iloc[i].split(' ')) == 1 for i in range(len(wikidata_df))]]
    wikidot_df.reset_index(drop=True)

    for i in range(len(wikidot_df)):
        T = [token.lemma_ for token in nlp(wikidot_df.iloc[i]['Title']) if token.lemma_ not in fr_stop]
        if len(T)>0 :
            wikidot_df.iloc[i]["Title"] = T[0]
        else : 
            wikidot_df.iloc[i]["Title"] = "" 

    noyaux_dict = {wikidot_df.iloc[i]["Title"]: wikidot_df.iloc[i]["Fonctions"] for i in range(len(wikidot_df)) if 
                    wikidot_df.iloc[i]["Title"] != ""}
    for noyau in set(noyaux_dict.values()):
        data_df[noyau] = [dictionnize(data_df.iloc[i]['text'],noyau,noyaux_dict) for i in range(len(data_df))]
    index_to_cut = []
    for i in range(len(data_df)): 
        if min([len(data_df.iloc[i][col]) for col in set(noyaux_dict.values())])==0:
            index_to_cut.append(i)

    data_df = data_df.iloc[[i for i in range(len(data_df)) if i not in index_to_cut]]
    data_df = data_df.reset_index(drop=True)

    train_test_ratio = 0.90
    train_valid_ratio = 8/9

    df_full_train, df_test = train_test_split(data_df, train_size = train_test_ratio, random_state = 1)
    df_train, df_valid = train_test_split(df_full_train, train_size = train_valid_ratio, random_state = 1)

    df_train.to_csv(args.data_path + '/train_lem_noyaux.csv',sep=',', index=False)
    df_valid.to_csv(args.data_path + '/valid_lem_noyaux.csv',sep=',', index=False)
    df_test.to_csv(args.data_path + '/test_lem_noyaux.csv',sep=',', index=False)


if __name__ == "__main__":
    main()