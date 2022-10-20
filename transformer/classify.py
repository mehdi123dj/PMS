# -*- coding: utf-8 -*-
import json
import logging
import os
import argparse

import numpy as np
import pandas as pd 
from farm.infer import Inferencer

from utils import entropy
from sklearn.metrics import precision_score,recall_score,accuracy_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = '../data'

class Classifier:
    def __init__(self, model_path):
        """
        Parameters:
        model_path: path toward the stored model
        """
        self.model = Inferencer.load(model_path, return_class_probs=True,batch_size=64)

    def run(self, content_list):
        """
        Parameters:
        content_list: a list of dictionaries  of the form :
        [{text : content_1},{text : content_2}...{text : content_N}]

        Output:
        the corresponding list of predictions on the
        input set of texts (labels and probabilities)
        """
        result = self.model.inference_from_dicts(dicts=content_list)
        self.model.close_multiprocessing_pool()
        return result


class Classify():
    """
    Push a class for each document found in the index.
    """

    def __init__(self,model_path,entropy_basis):
        super().__init__()

        # load the two models to classify the files
        with open(model_path + "/processor_config.json") as json_file:
            self.label_list = json.load(json_file)["tasks"]["text_classification"][
                "label_list"
            ]
        self.classifier = Classifier(model_path)
        self.entropy_basis = entropy_basis

    def process_document(self, paragraph):

        # Classify the file by a trained model
        result = self.classifier.run([{"text": paragraph}])
        P_list = [result[0]["predictions"][0]["probability"]][0]

        # Get the classes and their scores 
        label_predicted = self.label_list[np.argmax(P_list)]
        entropy_score = entropy(P_list, self.entropy_basis)
        best_score = max(P_list)
        return label_predicted,best_score,entropy_score 

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-mp",
        "--model-path",
        help="The location of the model",
        default=os.path.join(MODELS_DIR, "classifier"),
    )

    parser.add_argument(
        "-d",
        "--data-path",
        help="The location of the data to test",
        default=os.path.join(DATA_DIR, "test.tsv"),
    )

    parser.add_argument(
        "-eb",
        "--entropy-basis",
        help="The basis for the entropy calculation",
        default=2,
    )
    args = parser.parse_args()
    logging.getLogger("farm").setLevel(level=logging.ERROR)

    results = dict()
    classif = Classify(args.model_path,args.entropy_basis)

    test_docs = pd.read_csv(args.data_path,sep = "\t")

    for i in range(len(test_docs)): 
        doc_txt,label = test_docs.iloc[i]["text"],test_docs.iloc[i]["label"]

        label_predicted,best_score,entropy_score  = classif.process_document(doc_txt)
        results[doc_txt] = (label,label_predicted,best_score,entropy_score)

    y_true = [l[0] for l in results.values()]
    y_pred = [int(l[1]) for l in results.values()]

    test_docs['pred'] = y_pred
    test_docs.to_csv(os.path.join(DATA_DIR,'predicted_'+args.model_path.split('/')[-1]+'.csv'))
    print("Precision : ",precision_score(y_true,y_pred,average ="weighted"))
    print("Recall : ",recall_score(y_true,y_pred,average ="weighted"))
    print("Accuracy : ",accuracy_score(y_true,y_pred))


if __name__ == "__main__":
    main()

