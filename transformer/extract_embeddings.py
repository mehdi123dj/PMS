# -*- coding: utf-8 -*-
import logging
import os

from farm.infer import Inferencer

from utils import clean_text



class Embedder:
    def __init__(self, model_path):
        """
        Parameters:
        model_path: path toward the stored model
        Chose the model from which we are going to
        compute the embeddings
        """
        self.model = Inferencer.load(model_path)

    def run(self, content_list):
        """
        Parameters:
        content_list : a list of dictionaries  of the form :
        [{text : content_1},{text : content_2}...{text : content_N}]

        Output:
        the corresponding list of predictions on the
        input set of texts (labels and probabilities)
        """
        result = self.model.extract_vectors(dicts=content_list)
        self.model.close_multiprocessing_pool()
        return result


class ExtractEmbeddings():
    """
    Push an embedding for each document found in the index.
    """

    def __init__(self,model_path):
        super().__init__()

        # load the model
        self.embedding = Embedder(model_path)

    def process_document(self, text_content):

        content = clean_text(text_content)
        content_list = [
            {"text": content}        
            ]

        # then compute the embedding of the whole file 

        vector = self.embedding.run(content_list)[0]["vec"]
        return vector


