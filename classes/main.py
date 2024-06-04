from fastapi import FastAPI
from pydantic import BaseModel
import random
from scipy.sparse import csr_matrix

from classes.query_processing import QueryProcessing
from classes.utils import antique_output_file, wikir_output_file
from matching_ranking import MatchingRanking
import pickle

# import nltk
# nltk.download('punkt')

app = FastAPI()


class SearchRequest(BaseModel):
    datasetName: str
    query: str


@app.get("/search/data/")
def greet(datasetName: str, query: str):
    # Preprocess and vectorize the query using the loaded vectorizer

    # Load the document corpus
    doc_file = antique_output_file if datasetName == "antique" \
        else wikir_output_file if datasetName == "wikir" else None

    if not doc_file:
        return {"error": "Invalid dataset name. Please use 'antique' or 'wikir'."}

        # Create a Matching_Ranking object
    matcher = MatchingRanking(doc_file)
    preprocessed_query = matcher.preprocess_query(query)
    query_vector = matcher.data_rep.vectorizer.transform([preprocessed_query.split("\t", 1)[1]])

    # Convert the query vector to a sparse matrix
    query_vector = csr_matrix(query_vector)

    doc_vectors = matcher.data_rep.vsm
    # Call the matching function {{ Cos }}
    related_doc_id, related_documents = matcher.match(query_vector, doc_vectors)

    query_id = preprocessed_query.split("\t", 1)[0]

    process = QueryProcessing()
    query_vector = process.process_query(query_id, query)

    # Save the results to a file

    matcher.save_results(query_id, related_doc_id, related_documents)

    response = {
        "related_documents": related_documents  # Return the related documents
    }

    return response


greet('antique', 'Why do we need to tell customer service operator about our sickness in order to make appointment w/ a about doctor?')
