from utils import getText, clean_text, segmente, store_data, save_checkpoint, save_metrics , proba_to_class
import argparse
from os.path import isfile, join
from os import listdir
import pandas as pd 
import torchtext as tt 
from torchtext import data
from torchtext import datasets
import fr_core_news_md
import torch
import torch.nn as nn
from datetime import datetime
import os 
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from model import Multi_LSTM
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score

def train(model,
          optimizer,
          criterion,
          train_loader,
          valid_loader,
          num_epochs,
          eval_every,
          file_path,
          best_valid_loss,
          device):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for it in train_loader:    
            
            labels = it.noyau.long() -1
            text,text_len = it.text
            G,G_len = it.G
            S,S_len = it.S
            P,P_len = it.P
            
            Lens_s = S_len 
            Lens_p = P_len 
            Lens_g = G_len
            Lens_t = text_len 

            labels = labels.to(device)

            text = text.to(device)
            G = G.to(device)
            S = S.to(device)
            P = P.to(device)

            Lens_s = Lens_s.to("cpu")
            Lens_p = Lens_p.to("cpu")
            Lens_g = Lens_g.to("cpu")
            Lens_t = Lens_t.to("cpu")

            output = model(text, G, S, P,Lens_g,Lens_s,Lens_p,Lens_t)
            
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    
                    # validation loop

                    for itv in valid_loader:    
                        labels_v = itv.noyau.long() -1
                        text_v,text_len_v = itv.text   
                        G_v,G_len_v = itv.G
                        S_v,S_len_v = itv.S
                        P_v,P_len_v = itv.P

                        Lens_vg = G_len_v.to("cpu") 
                        Lens_vs = S_len_v.to("cpu")
                        Lens_vp = P_len_v.to("cpu")
                        Lens_vt = text_len_v.to("cpu")


                        output_v = model(text_v, G_v, S_v, P_v,Lens_vg,Lens_vs,Lens_vp,Lens_vt)

                        loss_v = criterion(output_v, labels_v)
                        valid_running_loss += loss_v.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

def test(model,test_iterator):
    model.eval()
    preds = []
    labs = []
    with torch.no_grad():                    
        for it in test_iterator:    
            labels = it.noyau.long() -1
            labs.extend(labels.tolist())
            text,text_len = it.text   
            G,G_len = it.G
            S,S_len = it.S
            P,P_len = it.P
            Lens_g = G_len
            Lens_s = S_len 
            Lens_p = P_len
            Lens_t = text_len

            Lens_g = Lens_g.to("cpu")
            Lens_s = Lens_s.to("cpu")
            Lens_p = Lens_p.to("cpu")
            Lens_t = Lens_t.to("cpu")

            output = model(text, G, S, P,Lens_g,Lens_s,Lens_p,Lens_t)
            predicted = proba_to_class(output)
            preds.extend(predicted)
    return precision_score(labs,preds,average = "macro"),recall_score(labs,preds,average = "macro")


def main():
    """
    Collect arguments and run.
    """
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-dp",
        "--data-path",
        help="The path to the paragraphs data set",
        default="../data",
    )

    parser.add_argument(
        "-mp",
        "--model-path",
        help="The path to store models",
        default=os.path.join("./models/multi_sequence/", f"trained_classifier_{date}"),
    )

    parser.add_argument(
        "-ne",
        "--num-epoch",
        help="The number of epochs",
        default=5,
        type = int, 
    )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        help="The learning rate",
        default=0.001,
        type = float, 
    )

    parser.add_argument(
        "-bs",
        "--batch-size",
        help="The batch size",
        default=1,
        type = float, 
    )

    parser.add_argument(
        "-nt",
        "--num-lstm-t",
        help="The number of lstm layers in the text sequence",
        default=1,
        type = int, 
    )

    parser.add_argument(
        "-nf",
        "--num-lstm-f",
        help="The number of lstm layers in functions sequences",
        default=1,
        type = int, 
    )

    parser.add_argument(
        "-edt",
        "--embed-dim-t",
        help="The dimension of the embedding of the text tokens",
        default=16,
        type = int, 
    )


    parser.add_argument(
        "-edf",
        "--embed-dim-f",
        help="The dimension of the embedding of the function tokens",
        default=16,
        type = int, 
    )

    parser.add_argument(
        "-hdt",
        "--hidden-dim-t",
        help="The dimension of the hidden layer in the text encoding",
        default=8,
        type = int, 
    )

    parser.add_argument(
        "-hdf",
        "--hidden-dim-f",
        help="The dimension of the hidden layer in the function encoding",
        default=8,
        type = int, 
    )

    parser.add_argument(
        "-dr",
        "--drop",
        help="The dropout probabilty",
        default=.1,
        type = float, 
    )

    args = parser.parse_args()
    os.mkdir(args.model_path)

    TEXT = data.Field(tokenize='spacy',
                    tokenizer_language = "fr_core_news_md",
                    lower=True, 
                    include_lengths=True,
                    stop_words = [],
                    batch_first=True)

    G_FIELD = data.Field(tokenize='spacy',
                    tokenizer_language = "fr_core_news_md",
                    lower=True, 
                    include_lengths=True,
                    stop_words =[],
                    batch_first=True)

    S_FIELD = data.Field(tokenize='spacy',
                    tokenizer_language = "fr_core_news_md",
                    lower=True, 
                    include_lengths=True,
                    stop_words = [],
                    batch_first=True)

    P_FIELD = data.Field(tokenize='spacy',
                    tokenizer_language = "fr_core_news_md",
                    lower=True, 
                    include_lengths=True,
                    stop_words = [],
                    batch_first=True)

    NOYAU = data.Field(sequential=False,
                    use_vocab=False,
                    batch_first=True,
                    dtype=torch.float,
                    is_target = True)

    Fields = [('noyau', NOYAU), ('text', TEXT),('G',G_FIELD),('S',S_FIELD),('P',P_FIELD)]

    train_data, valid_data, test_data = TabularDataset.splits(path=args.data_path, 
                                            train='train_lem_stop.csv', 
                                            validation='valid_lem_stop.csv',
                                            test='test_lem_stop.csv',
                                            format='CSV',
                                            fields=Fields,
                                            skip_header=True)
                                            
    TEXT.build_vocab(train_data)
    G_FIELD.build_vocab(train_data)
    S_FIELD.build_vocab(train_data)
    P_FIELD.build_vocab(train_data)
    setattr(G_FIELD,"vocab",TEXT.vocab)
    setattr(S_FIELD,"vocab",TEXT.vocab)
    setattr(P_FIELD,"vocab",TEXT.vocab)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = args.batch_size 

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        sort_key = lambda x: len(x.text), #sort by text 
        batch_size=BATCH_SIZE,
        device=device)

    model = Multi_LSTM(vocab_size = len(TEXT.vocab),
                            hidden_t = args.hidden_dim_t, 
                            hidden_f = args.hidden_dim_f,
                            embed_dim_t = args.embed_dim_t,
                            embed_dim_f = args.embed_dim_f,
                            nlstm_t = args.num_lstm_t,
                            nlstm_f = args.num_lstm_f,
                            dropout = args.drop,
                            ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train(model=model,
        optimizer=optimizer, 
        criterion = nn.CrossEntropyLoss(),
        train_loader = train_iterator,
        valid_loader = valid_iterator,
        num_epochs=args.num_epoch,
        eval_every = len(train_iterator) // 2,
        file_path = args.model_path,
        best_valid_loss = float("Inf"),
        device = device)
    
    precision, recall = test(model,test_iterator)

    
    print('test precision : ', precision)
    print('test recall : ', recall)
    df_save = pd.DataFrame.from_dict({'scores':[precision,recall]})
    df_save.to_csv(os.path.join(args.model_path,'scores.csv'))


if __name__ == "__main__":
    main()

