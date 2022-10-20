# -*- coding: utf-8 -*-
import argparse
import csv
import logging
import os as os
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from utils import clean_text
from farm.train import Trainer


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

#DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_DIR = '../data'
TRAIN_FILE_NAME = "train.tsv"
TEST_FILE_NAME = "test.tsv"


def train_classifier(
    model_name_or_path,
    data_path,
    output_path,
    labellist,
    max_length,
    n_epoch,
    metric,
    batch_size,
    n_gpu,
    learning_rate,
    dropout_prob,
    evaluate_every,
):
    """
    This function saves a language model and a processor in the output_path that are
    later used for the embedding extraction and the classification.

    Parameters :
    model_name_or_path: the language model to select
    data_path and output_path: the directories in which
    we stored the data (generated with the get_training_data procedure)
    labellist: list of all the labels in the data set
    max_length: the maximum number of tokens processed by the language model
    n_epoch: the number of epochs
    metric:  the score shown during the training
    batch_size: the batch size
    n_gpu: the number of gpus available for training and evaluation (if this device is available)
    learning_rate: the learning rate
    dropout_prob: the dropout probability
    evaluate_every: evaluate the metric each number of examples
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Devices available: {}".format(device))
    
    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(pretrained_model_name_or_path=model_name_or_path)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    processor = TextClassificationProcessor(
        tokenizer=tokenizer,
        max_seq_len=max_length,
        data_dir=data_path,
        train_filename=TRAIN_FILE_NAME,
        test_filename=TEST_FILE_NAME,
        label_list=labellist,
        metric=metric,
        label_column_name="label",
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a
    # few descriptive statistics of our datasets
    data_silo = DataSilo(processor=processor, batch_size=batch_size)
    # 4. Create an AdaptiveMode
    # a) which consists of a pretrained language model as a basis
    MODEL_NAME_OR_PATH = model_name_or_path
    
    language_model = LanguageModel.load(MODEL_NAME_OR_PATH)
    #language_model = LanguageModel.load(pretrained_model_name_or_path = 'camembert-base')
    
    # b) and a prediction head on top that is suited for our task => Text classification
    prediction_head = TextClassificationHead(
        num_labels=len(labellist),
        class_weights=data_silo.calculate_class_weights(
            task_name="text_classification"
        ),
    )
    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=dropout_prob,
        lm_output_types=["per_sequence"],
        device=device,
    )

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epoch,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epoch,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device,
    )
    # 7. Let it grow
    model = trainer.train()
    # Then store it
    model.save(output_path)
    processor.save(output_path)


def get_training_data(data_path, max_length):
    """
    Get documents classes from a CSV file (containing the name and the annotated_label
    of each file) and documents contents from an ElasticSearch instance to create training data.
    This file is provided together with the data, and is not generated via the elasticsearch instance.
    Parameters :
    host and index to conntect to the ES instance
    data_path to get the annotations of the files
    max_length: the maximum number of tokens processed by the language model

    Returns the full data set in a dictionary of the form : {file_name: [content,annotated_class]}
    """
    # Download the data set

    datadict = dict()
    with open(os.path.join(data_path,"paragraphs.csv")) as f:
        reader = csv.reader(f, delimiter=",")
        for i,line in enumerate(reader):
            t = line[0].replace('"','')
            datadict[line[0]] = line[1]
    return datadict


def split_dataset(data, ratio, data_path):
    """
    This function splits the data set and stores two files :
    train file and test file  in data_path

    Parameters : data, a dictionary of the same form that is returned by the
    get_training_data function.
    ratio : determines the train-test ratios
    data_path: the folder in which to put the train file and test file
    """
    N = len(data)
    N_train = int(N * ratio)
    train = list(np.random.choice(list(data.keys()), size=N_train, replace=False))
    test = list(set(data.keys()) - set(train))


    df_train = pd.DataFrame.from_dict({'text': [clean_text(t) for t in train],
                'label': [data[t] for t in train]})
    df_train.to_csv(os.path.join(data_path,TRAIN_FILE_NAME),sep = '\t')


    df_train = pd.DataFrame.from_dict({'text': [clean_text(t) for t in test],
                'label': [data[t] for t in test]})
    df_train.to_csv(os.path.join(data_path,TEST_FILE_NAME),sep = '\t')




def run(
    data_path,
    ratio,
    model_name_or_path,
    output_path,
    max_length,
    n_epoch,
    metric,
    batch_size,
    n_gpu,
    learning_rate,
    dropout_prob,
    evaluate_every,
):
    """
    Launches the following process :
    Reads the data from ES, collects the labels, splits the data set into train and test
    then stores both in data_path.Launches the training of the classifier, then stores the resulting model.

    Parameters :
    The inputs of the train_classifier function : detailed below train_classifier
    host and index : to connect to the ES instance
    ratio : the ratio of the train dataset in comparison with the full data set
    """

    data = get_training_data(data_path, max_length)


    labellist = list(data.values())

    #split_dataset(data, ratio, data_path)
    train_classifier(
        model_name_or_path,
        data_path,
        output_path,
        labellist,
        max_length,
        n_epoch,
        metric,
        batch_size,
        n_gpu,
        learning_rate,
        dropout_prob,
        evaluate_every,
    )


def main():
    """
    Collect arguments and run.
    """
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "-d",
        "--data-path",
        help="The folder containing the annotation file as well as the train file and test file files once they are generated ",
        default=DATA_DIR,
    )

    parser.add_argument(
        "-r",
        "--ratio",
        help="the ratio of the training data set (reported to the full data set)",
        default=0.8,
        type = float 
    )

    parser.add_argument(
        "-mnp",
        "--model-name-or-path",
        help="the name of one of the models registered in FARM, or the path of a pre-trained model to reuse.",
        #default=os.path.join(MODELS_DIR, "camembert/camembert-large"),
        default="camembert-base",
    )

    parser.add_argument(
        "-out",
        "--output-path",
        help="the location in which to save the model ",
        default=os.path.join(MODELS_DIR, f"trained_classifier_{date}"),
    )

    parser.add_argument(
        "-mxl",
        "--max-length",
        help="the maximum length of the input supported by the language model",
        default=512,
        type = int 
    )

    parser.add_argument(
        "-ne",
        "--n-epoch",
        help="the number of epochs used for training",
        default=12,
        type = int 
        )

    parser.add_argument(
        "-met",
        "--metric",
        help="name of metric that shall be used for evaluation “acc” or “f1_macro”.  ",
        choices=["acc", "f1_macro"],
        default="acc",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        help="the size of the batch",
        default=16,
        type = int 
    )
    parser.add_argument(
        "-ngpu",
        "--n-gpu",
        help="the number of gpus available for training and evaluation.",
        default=1,
        type = int
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        help="the learning rate",
        default=2e-5,
        type = float 
    )
    parser.add_argument(
        "-drop",
        "--dropout-prob",
        help="The dropout rate",
        default=0.1,
        type = float 
    )
    parser.add_argument(
        "-eval",
        "--evaluate-every",
        help="The number of examples to attain before evaluation",
        default=500,
        type = int 
    )

    #logging.getLogger("farm").setLevel(level=logging.ERROR)

    args = parser.parse_args()
    run(
        args.data_path,
        args.ratio,
        args.model_name_or_path,
        args.output_path,
        args.max_length,
        args.n_epoch,
        args.metric,
        args.batch_size,
        args.n_gpu,
        args.learning_rate,
        args.dropout_prob,
        args.evaluate_every,
    )


if __name__ == "__main__":
    main()
