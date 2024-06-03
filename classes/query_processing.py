from classes.utils import query_output_file, query_tfidf_file
from text_processing import TextProcessing
from representation_data import DataRepresentation  # Import DataRepresentation
import joblib  # Import joblib for binary saving
import numpy as np


class QueryProcessing:
    def process_query(self, user_query):
        # Processes a user query using the TextProcessing class and generates a TF-IDF vector.
        # Args:
        # user_query (str): The user's query string.
        # output_file (str, optional): The path to the output TSV file. Defaults to 'query_output.tsv'.
        # dataset_name (str, optional): The name of the dataset to use for vectorization. Defaults to 'antique'.
        # Returns:
        # tuple: A tuple containing the path to the output file and the TF-IDF vector of the query.

        processor = TextProcessing()
        df = processor.process_text(user_query)
        df.to_csv(query_output_file, sep='\t', index=False)

        # Create DataRepresentation object for the processed query
        data_rep = DataRepresentation(query_output_file)  # Use the output file as input
        data_rep.create_vsm()  # Calculate TF-IDF for the query

        # Get the vocabulary and normalize it
        vocabulary = [word.lower() for word in data_rep.get_vocabulary()]

        # Create a query vector with the correct dimensions
        query_vector = np.zeros(len(vocabulary))

        # Extract query words from the processed DataFrame
        query_words = df['tokens'].tolist()  # Get the list of tokens from the DataFrame
        query_words = [word.lower() for word in query_words]  # Normalize to lowercase

        # Populate the query vector based on your query words
        for i, word in enumerate(vocabulary):
            if word in query_words:
                query_vector[i] = 1

        # Save the query vector to a file in the specified path (binary format)
        joblib.dump(query_vector, query_tfidf_file)  # Save in binary format

        return query_tfidf_file, query_vector


# Example usage:
process = QueryProcessing()
query_vector = process.process_query("small group like the is was are ## ?")
print(f"Output file: {query_tfidf_file}")
print(f"Query vector: {query_vector}")
