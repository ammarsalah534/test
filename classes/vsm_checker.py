import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VSMChecker:
    def __init__(self, processed_text_file, saved_results_file):
        """
        Initializes the VSMChecker object.

        Args:
            processed_text_file (str): Path to the file containing processed text data.
            saved_results_file (str): Path to the file containing the saved TF-IDF vectors and vocabulary.
        """
        self.processed_text_file = processed_text_file
        self.saved_results_file = saved_results_file

        # self.load_data()
        self.load_results()

    # def load_data(self):
    #     """
    #     Loads the processed text data from the specified file.
    #     """
    #     self.processed_text_df = pd.read_csv(self.processed_text_file)
    #     # self.processed_text = self.processed_text_df['tokens'].tolist()  # Assuming 'words' is the column name

    def load_results(self):
        """
        Loads the saved TF-IDF vectors and vocabulary from the CSV file.
        """
        self.results_df = pd.read_csv(self.saved_results_file)
        self.vocabulary = self.results_df.columns.tolist()  # Get vocabulary from column names

        # Create a new TfidfVectorizer with the loaded vocabulary
        self.loaded_vectorizer = TfidfVectorizer(vocabulary=self.vocabulary)

    def calculate_similarity(self, doc1_index, doc2_index):
        """
        Calculates cosine similarity between two documents.

        Args:
            doc1_index (int): Index of the first document.
            doc2_index (int): Index of the second document.

        Returns:
            float: Cosine similarity score.
        """
        # Get the TF-IDF vectors from the results DataFrame
        tfidf_vectors = self.results_df.values

        # Reshape the vectors to be 2D arrays
        doc1_vector = tfidf_vectors[doc1_index].reshape(1, -1)  # Reshape to (1, n_features)
        doc2_vector = tfidf_vectors[doc2_index].reshape(1, -1)  # Reshape to (1, n_features)

        similarity = cosine_similarity(doc1_vector, doc2_vector)
        return similarity[0][0]  # Extract the single similarity value

# Example usage:
checker = VSMChecker('D:/ir_final_final_final_the_flinalest/data/antiqe_output/output_collection.tsv', 'D:/ir_final_final_final_the_flinalest/data/antiqe_output/query_tfidf_results.pkl')

doc1_index = 0
doc2_index = 0
similarity = checker.calculate_similarity(doc1_index, doc2_index)
print(f"Cosine Similarity between documents {doc1_index} and {doc2_index}: {similarity}")