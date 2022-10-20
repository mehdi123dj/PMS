from os import listdir
import pandas as pd 
from os.path import isfile, join
import re 
import docx
import csv
import math
import torch 
import numpy as np
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

def token_list_to_text(token_list):
    text = ''
    for t in token_list:
        text+=t + ' '
    return text[:-1]


def functionize(text,functions_dict,func):
    if func !='NA':
        List = [ t for t in text.split(' ') if t in functions_dict and functions_dict[t] ==func]
    else : 
        List = [ t for t in text.split(' ') if t not in functions_dict]
    return token_list_to_text(List)


def save_checkpoint(save_path, model, optimizer, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

def labels_to_tensors(labels,tensor_size):
    lab= labels.long().tolist()
    tensors = torch.zeros(labels.shape[0],tensor_size)
    for i in range(len(lab)):
        l = lab[i]
        T = torch.zeros(tensor_size)
        T[l-1]+=1
        tensors[i] += T
    return tensors.long()

def proba_to_class(output):
    pred = []
    for T in output:
        pred.append(T.argmax().item())
    return pred
    
def dictionnize(text,noyau,expressions_dict):
    if text != np.nan : 
        List = [ t for t in text.split(' ') if t in expressions_dict  and expressions_dict[t] == noyau]
        return token_list_to_text(List)
    else : 
        return text 