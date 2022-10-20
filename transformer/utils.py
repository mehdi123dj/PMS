from os import listdir
import pandas as pd 
from os.path import isfile, join
import re 
import docx
import csv
import math

def entropy(Liste, base):
    """
    This function computes the entropy of a list of probabilities, provided a basis.
    """
    if base == "natural":
        return -sum([p * math.log(p) for p in Liste])
    else:
        return (-1.0 / math.log(base)) * sum([p * math.log(p) for p in Liste])


def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

def clean_text(text):
    
    text = text.replace('\n',' ')
    text = text.replace('\t',' ')

    alphanumeric = re.findall("[^œa-zA-ZÀ-ÿ0-9'\-]",text)
    for an in alphanumeric:
        text = text.replace(an,' ')
    R = re.findall("[ \t]+",text)
    if len(R)>0:
        while set(R) != {' '} :  
            R = re.findall("[ \t]+",text)
            for r in R:
                text = text.replace(r,' ')
    return text

def segmente(txt):
    chopped = []
    ex = [e for e in set(re.findall(r"\[[0-9_]+\]", txt)) if len(e)==8]
    labs = [e for e in (re.findall(r"\[[0-9_]+\]", txt)) if len(e)==8]
    start_id = []
    for e in ex : 
        E = e.replace('[','\[')
        E = E.replace(']','\]')
        start_id.extend([ m.start() for m in re.finditer(E, txt)])
    start_id.append(len(txt))
    start_id = sorted(start_id)
    for i in range(len(start_id)-1): 
        start = start_id[i]+8
        finish = start_id[i+1]
        chopped.append(txt[start:finish])
    return [(clean_text(chopped[i]),labs[i][1:-1]) for i in range(len(chopped))]

def store_data(code_dict,path):
    with open(join(path,'paragraphs.csv'), 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        for k in code_dict.keys():
            datawriter.writerow([k,int(code_dict[k][0]),code_dict[k]])

def get_split(token_list, n_tokens, n_overlap):
    """
    This is used in train_classifier.py, classify.py and extract_embedding.py to
    split a document into a set of overlapping inputs, each of them having a fixed
    number of tokens. This number is the maximum supported by the language model
    used in the classifier.
    Parameters:
    token_list: the list of tokens to be processed
    n_tokens: the number of tokens by by subset in the returned list
    n_overlap : the number of overlapping tokens between the end of the
    previous list and the beginning of the following one.

    Output : a list of subsets of tokens
    """
    l_total = []
    l_partial = token_list[:n_tokens]
    if len(token_list) // (n_tokens - n_overlap) > 0:
        n = len(token_list) // (n_tokens - n_overlap)
    else:
        n = 1
    for w in range(1, n):
        l_partial = token_list[
            w * (n_tokens - n_overlap) : w * (n_tokens - n_overlap) + n_tokens
        ]
        l_total.append(" ".join(l_partial))
    return l_total