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
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from model import noyaux_LSTM
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
            
            labels = it.label.long() -1
            text,text_len = it.text
            gsp,gsp_len = it.gsp
            gps,gps_len = it.gps
            spg,spg_len = it.spg
            sgp,sgp_len = it.sgp
            pgs,pgs_len = it.pgs
            psg,psg_len = it.psg

            Lens_gps = gps_len
            Lens_gsp = gsp_len 
            Lens_spg = spg_len 
            Lens_sgp = sgp_len 
            Lens_pgs = pgs_len 
            Lens_psg = psg_len 
            Lens_t = text_len 

            labels = labels.to(device)
            text = text.to(device)
            gps = gps.to(device)
            gsp = gsp.to(device)
            spg = spg.to(device)
            sgp = sgp.to(device)
            psg = psg.to(device)
            pgs = pgs.to(device)

            Lens_gps = Lens_gps.to("cpu")
            Lens_gsp = Lens_gsp.to("cpu") 
            Lens_spg = Lens_spg.to("cpu") 
            Lens_sgp = Lens_sgp.to("cpu") 
            Lens_pgs = Lens_pgs.to("cpu") 
            Lens_psg = Lens_psg.to("cpu") 
            Lens_t = Lens_t.to("cpu")


            output = model(text,gsp, gps,sgp, spg,pgs, psg,Lens_t,Lens_gsp, Lens_gps,Lens_sgp ,Lens_spg,Lens_pgs,Lens_psg)
            
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
                        labels_v = itv.label.long() -1
                        text_v,text_len_v = itv.text
                        gsp_v,gsp_len_v = itv.gsp
                        gps_v,gps_len_v = itv.gps
                        spg_v,spg_len_v = itv.spg
                        sgp_v,sgp_len_v = itv.sgp
                        pgs_v,pgs_len_v = itv.pgs
                        psg_v,psg_len_v = itv.psg

                        Lens_gps_v = gps_len_v.to("cpu") 
                        Lens_gsp_v = gsp_len_v.to("cpu") 
                        Lens_spg_v = spg_len_v.to("cpu") 
                        Lens_sgp_v = sgp_len_v.to("cpu") 
                        Lens_pgs_v = pgs_len_v.to("cpu") 
                        Lens_psg_v = psg_len_v.to("cpu") 
                        Lens_t_v = text_len_v.to("cpu") 

                        output_v = model(text_v,gsp_v, gps_v,sgp_v, spg_v,pgs_v, psg_v,Lens_t_v,Lens_gsp_v, Lens_gps_v,Lens_sgp_v ,Lens_spg_v,Lens_pgs_v,Lens_psg_v)

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

def test(model,test_iterator,device):
    model.eval()
    preds = []
    labs = []
    with torch.no_grad():                    
        for it in test_iterator:    
            labels = it.label.long() -1
            text,text_len = it.text
            gsp,gsp_len = it.gsp
            gps,gps_len = it.gps
            spg,spg_len = it.spg
            sgp,sgp_len = it.sgp
            pgs,pgs_len = it.pgs
            psg,psg_len = it.psg

            Lens_gps = gps_len
            Lens_gsp = gsp_len 
            Lens_spg = spg_len 
            Lens_sgp = sgp_len 
            Lens_pgs = pgs_len 
            Lens_psg = psg_len 
            Lens_t = text_len 

            labels = labels.to(device)
            text = text.to(device)
            gps = gps.to(device)
            gsp = gsp.to(device)
            spg = spg.to(device)
            sgp = sgp.to(device)
            psg = psg.to(device)
            pgs = pgs.to(device)

            Lens_gps = Lens_gps.to("cpu")
            Lens_gsp = Lens_gsp.to("cpu") 
            Lens_spg = Lens_spg.to("cpu") 
            Lens_sgp = Lens_sgp.to("cpu") 
            Lens_pgs = Lens_pgs.to("cpu") 
            Lens_psg = Lens_psg.to("cpu") 
            Lens_t = Lens_t.to("cpu")


            output =  model(text,gsp, gps,sgp, spg,pgs, psg,Lens_t,Lens_gsp, Lens_gps,Lens_sgp ,Lens_spg,Lens_pgs,Lens_psg)
            
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
        default=os.path.join("./models/double_sequence/", f"trained_classifier_{date}"),
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
                    stop_words = fr_stop,
                    batch_first=True)

    gsp_FIELD = data.Field(tokenize='spacy',
                    tokenizer_language = "fr_core_news_md",
                    lower=True, 
                    include_lengths=True,
                    stop_words = fr_stop,
                    batch_first=True)

    gps_FIELD = data.Field(tokenize='spacy',
                    tokenizer_language = "fr_core_news_md",
                    lower=True, 
                    include_lengths=True,
                    stop_words = fr_stop,
                    batch_first=True)

    sgp_FIELD = data.Field(tokenize='spacy',
                    tokenizer_language = "fr_core_news_md",
                    lower=True, 
                    include_lengths=True,
                    stop_words = fr_stop,
                    batch_first=True)

    spg_FIELD = data.Field(tokenize='spacy',
                    tokenizer_language = "fr_core_news_md",
                    lower=True, 
                    include_lengths=True,
                    stop_words = fr_stop,
                    batch_first=True)

    pgs_FIELD = data.Field(tokenize='spacy',
                    tokenizer_language = "fr_core_news_md",
                    lower=True, 
                    include_lengths=True,
                    stop_words = fr_stop,
                    batch_first=True)

    psg_FIELD = data.Field(tokenize='spacy',
                    tokenizer_language = "fr_core_news_md",
                    lower=True, 
                    include_lengths=True,
                    stop_words = fr_stop,
                    batch_first=True)

    NOYAU = data.Field(sequential=False,
                    use_vocab=False,
                    batch_first=True,
                    dtype=torch.float,
                    is_target = True)
    Fields = [('text', TEXT),('label', NOYAU),
    ('spg', spg_FIELD),('pgs', pgs_FIELD),('psg', psg_FIELD),
    ('gps', gps_FIELD),('sgp', sgp_FIELD),('gsp', gsp_FIELD)]


    train_data, valid_data, test_data = TabularDataset.splits(path=args.data_path, 
                                            train='train_lem_noyaux.csv', 
                                            validation='valid_lem_noyaux.csv',
                                            test='test_lem_noyaux.csv',
                                            format='CSV',
                                            fields=Fields,
                                            skip_header=True)

    TEXT.build_vocab(train_data)
    gsp_FIELD.build_vocab(train_data)
    gps_FIELD.build_vocab(train_data)
    sgp_FIELD.build_vocab(train_data)
    spg_FIELD.build_vocab(train_data)
    pgs_FIELD.build_vocab(train_data)
    psg_FIELD.build_vocab(train_data)

    setattr(gsp_FIELD,"vocab",TEXT.vocab)
    setattr(gps_FIELD,"vocab",TEXT.vocab)
    setattr(sgp_FIELD,"vocab",TEXT.vocab)
    setattr(spg_FIELD,"vocab",TEXT.vocab)
    setattr(pgs_FIELD,"vocab",TEXT.vocab)
    setattr(psg_FIELD,"vocab",TEXT.vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = args.batch_size 

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        sort_key = lambda x: len(x.text), #sort by text 
        batch_size=BATCH_SIZE,
        device=device)

    model = noyaux_LSTM(vocab_size = len(TEXT.vocab),
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
    
    #precision, recall = test(model,test_iterator,device)
    #print('test precision : ', precision)
    #print('test recall : ', recall)
    #df_save = pd.DataFrame.from_dict({'scores':[precision,recall]})
    #df_save.to_csv(os.path.join(args.model_path,'scores.csv'))


if __name__ == "__main__":
    main()

