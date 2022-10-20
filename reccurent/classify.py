from utils import load_checkpoint , proba_to_class
import argparse
from os.path import isfile, join
from model import single_seq_LSTM, double_seq_LSTM, Multi_LSTM
import pandas as pd 

def classify(model,model_type,data_iterator):
    model.eval()
    preds = []
    with torch.no_grad():                    
        for it in data_iterator:    
            text,text_len = it.text   
            G,G_len = it.G
            S,S_len = it.S
            P,P_len = it.P
            NA,NA_len = it.NA
            Lens_f = G_len+S_len+P_len + NA_len 
            Lens_t = text_len
            Lens = Lens_f+ Lens_t
            if model_type == 'single':
                output = model(text, G, S, P,NA,Lens)
            elif 
                model_type == 'double':
                output = model(text, G, S, P,NA,Lens_f,Lens_t)
            predicted = proba_to_class(output)
            preds.extend(predicted)
    return preds

def main():
    """
    Collect arguments and run.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dp",
        "--data-path",
        help="The path to the data to classify",
        default="../data",
    )
    
    parser.add_argument(
        "-mp",
        "--model-path",
        help="The path to the stored model to use",
        default="./models",
    )

    parser.add_argument(
        "-mp",
        "--model-path",
        help="The path to store models",
        default=os.path.join("./models/single_sequence/", f"trained_classifier_{date}"),
    )

    parser.add_argument(
        "-vp",
        "--vocab-path",
        help="The path to pickle vocab object",
        default="../data/vocab.pkl",
    )

    parser.add_argument(
        "-mt",
        "--model-type",
        help="The model type",
        default="single",
        choices = ['single','double']
    )


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_type == 'single':
        model = single_seq_LSTM(vocab_size = len(TEXT.vocab)).to(device)
    elif args.model_type == 'double':
        model = double_seq_LSTM(vocab_size = len(TEXT.vocab)).to(device)
    elif args.model_type == 'multi':
        model = Multi_LSTM(vocab_size = len(TEXT.vocab)).to(device)


    optimizer = optim.Adam(model.parameters())
    model = load_checkpoint(args.model_path,model,optimizer)

    TEXT = data.Field(tokenize='spacy',
                    tokenizer_language = "fr_core_news_md",
                    lower=True, 
                    include_lengths=True,
                    stop_words = fr_stop,
                    batch_first=True)

    G_FIELD = data.Field(tokenize='spacy',
                    tokenizer_language = "fr_core_news_md",
                    lower=True, 
                    include_lengths=True,
                    stop_words = fr_stop,
                    batch_first=True)

    S_FIELD = data.Field(tokenize='spacy',
                    tokenizer_language = "fr_core_news_md",
                    lower=True, 
                    include_lengths=True,
                    stop_words = fr_stop,
                    batch_first=True)

    P_FIELD = data.Field(tokenize='spacy',
                    tokenizer_language = "fr_core_news_md",
                    lower=True, 
                    include_lengths=True,
                    stop_words = fr_stop,
                    batch_first=True)


    NA_FIELD = data.Field(tokenize='spacy',
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

    Fields = [('noyau', NOYAU), ('text', TEXT),('G',G_FIELD),('S',S_FIELD),('P',P_FIELD),('NA',NA_FIELD)]

    data = TabularDataset(path=os.path.join(args.data_path,"test_lem.csv"), 
                                            format='CSV',
                                            fields=Fields,
                                            skip_header=True)

    TEXT.build_vocab(data)
    G_FIELD.build_vocab(train_data)
    S_FIELD.build_vocab(train_data)
    P_FIELD.build_vocab(train_data)
    NA_FIELD.build_vocab(train_data)
    vocab = pickle.load(args.vocab_path)
    setattr(TEXT,"vocab",vocab)
    setattr(G_FIELD,"vocab",vocab)
    setattr(S_FIELD,"vocab",vocab)
    setattr(P_FIELD,"vocab",vocab)
    setattr(NA_FIELD,"vocab",vocab)

    labels = pd.read_csv(os.path.join(args.data_path,"test_lem.csv"))
    preds = classify(model,model_type,data)
    pd.DataFrame.to_csv(os.path.join(args.data_path),arg.model_path)


if __name__ == "__main__":
    main()