import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from scipy.sparse import csr_matrix

from classes.utils import antique_output_file, antique_tfidf_file


class DataRepresentation:
    def __init__(self, data_file):
        # Initializes the DataRepresentation object.
        # Args:
        # data_file (str): Path to the file containing cleaned text data (one document per line).
        self.data_file = data_file
        self.df = pd.DataFrame({'text': self.read_cleaned_text(data_file)})  # Read cleaned text
        self.vectorizer = TfidfVectorizer()  # Create a TfidfVectorizer instance
        self.vsm = self.vectorizer.fit_transform(self.df['text'])  # Calculate VSM for the dataset
        self.vocabulary = self.vectorizer.get_feature_names_out()  # Store the vocabulary

    def read_cleaned_text(self, data_file):
        # Reads cleaned text data from the file, assuming one document per line.
        # Args:
        # data_file (str): Path to the file containing cleaned text data.
        # Returns:
        # list: A list of cleaned text documents.
        with open(data_file, 'r') as f:
            cleaned_texts = f.readlines()
        return [text.strip() for text in cleaned_texts]

    def create_vsm(self):
        # Creates a VSM (Vector Space Model) representation of the text data.
        # Returns:
        # scipy.sparse.csr_matrix: A sparse matrix representing the VSM.
        return self.vsm  # Return the pre-calculated VSM

    def get_tfidf_vectors(self):
        # Calculates TF-IDF vectors for the text data.
        # Returns:
        # scipy.sparse.csr_matrix: A sparse matrix of TF-IDF vectors.
        return self.vsm  # Return the pre-calculated VSM

    def get_vocabulary(self):
        # Returns the vocabulary used for the VSM.
        # Returns:
        # list: A list of unique words in the vocabulary.
        return self.vocabulary

    def save_results(self, output_file):
        # Save the sparse matrix and vocabulary to a binary file
        with open(output_file, 'wb') as f:
            pickle.dump((self.vsm, self.vocabulary), f)

    def load_results(self, input_file):
        with open(input_file, 'rb') as f:
            self.vsm, self.vocabulary = pickle.load(f)


# data_rep = DataRepresentation(antique_output_file)  # Replace with the path to your cleaned file
# data_rep.create_vsm()  # Calculate TF-IDF vectors
# data_rep.save_results(antique_tfidf_file)
# data_rep.load_results(antique_tfidf_file)
