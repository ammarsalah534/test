import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from classes.utils import antique_tfidf_file, related_output_file
from representation_data import DataRepresentation
from scipy.sparse import csr_matrix
from nltk.tokenize import word_tokenize
import json


class MatchingRanking:
    def __init__(self, doc):
        # Load the document corpus
        with open(doc, 'r', encoding='utf-8') as f:
            self.doc_corpus = f.readlines()

        # Create a DataRepresentation object
        self.data_rep = DataRepresentation(doc)

        # Save the fitted vectorizer for future use
        joblib.dump(self.data_rep.vectorizer, antique_tfidf_file)

    # Function to preprocess the query
    def preprocess_query(self, query):
        # Tokenize and preprocess the query here (e.g., lowercasing, removing stop words, etc.)
        tokens = word_tokenize(query.lower())
        # Implement any additional preprocessing if required
        return ' '.join(tokens)

    # Function to compute the cosine similarity and retrieve top document indices
    def match(self, query_vector, doc_vectors):
        # """
        # Calculates cosine similarity between a query vector and multiple document vectors.

        # Args:
        #     query_vector: The vector representing the query.
        #     doc_vectors: A list or array of document vectors.

        # Returns:
        #     A tuple containing:
        #         - related_doc_id: The indices of the top 10 most similar documents.
        #         - related_documents: The actual document texts.
        # """

        # Compute cosine similarity between the query vector and all document vectors
        cosine_similarities = cosine_similarity(doc_vectors, query_vector).flatten()

        # Get the indices of the top 10 most similar documents
        related_doc_id = cosine_similarities.argsort()[:-11:-1]

        # Extract the related documents from the corpus
        # related_documents = [self.doc_corpus[i].replace(" ", "\t", 1) for i in related_doc_id]
        related_documents = [self.doc_corpus[i] for i in related_doc_id]

        return related_doc_id, related_documents

    # Save the results to a file
    def save_results(self, query_id, related_doc_id):
        # Open the file in append mode

        with open(related_output_file, 'a') as f:
            for doc_id in related_doc_id:
                f.write(f"{query_id}\t{doc_id}\n")

        '''
        with open(related_output_file, 'a') as f:
            for index, document in zip(related_doc_id, related_documents):
                # f.write(str(index) + '\n')
                f.write(document)
                # f.write("-----------\n")
        '''
        print(f"Related document indices saved to {related_output_file}.")
