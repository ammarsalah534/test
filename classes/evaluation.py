import os

import numpy as np
import pandas as pd
from collections import defaultdict

from classes.utils import antique_related_file, wikir_related_file, antique_ground_truth_file, wikir_ground_truth_file


class Evaluation:
    def __init__(self, related_output_file, ground_truth_file=None):
        """
        Initializes the Evaluation class.

        Args:
            related_output_file (str): Path to the related output TSV file.
            ground_truth_file (str, optional): Path to the ground truth TSV file.
        """
        self.related_output = pd.read_csv(related_output_file, sep='\t', header=None,
                                          names=['query_id', 'index', 'doc_id', 'doc_text'])

        if ground_truth_file and os.path.isfile(ground_truth_file):
            self.ground_truth = pd.read_csv(ground_truth_file, sep='\t', header=None, names=['query_id', 'doc_id'])
        else:
            self.create_ground_truth_file('antique')
            self.ground_truth = pd.read_csv('ground_truth.tsv', sep='\t', header=None, names=['query_id', 'doc_id'])

    def create_ground_truth_file(self, dataset_name):
        """Creates a ground truth file based on the related output."""
        if dataset_name == 'antique':
            ground_truth_file = antique_ground_truth_file
            related_output = pd.read_csv(antique_related_file, sep='\t', header=None,
                                         names=['query_id', 'index', 'doc_id', 'doc_text'])
        elif dataset_name == 'wikir':
            ground_truth_file = wikir_ground_truth_file
            related_output = pd.read_csv(wikir_related_file, sep='\t', header=None,
                                         names=['query_id', 'index', 'doc_id', 'doc_text'])
        else:
            raise ValueError("Invalid dataset name. Please use 'antique' or 'wikir'.")

        ground_truth_data = []
        for query_id in related_output['query_id'].unique():
            query_docs = related_output[related_output['query_id'] == query_id]['doc_id'].tolist()
            for doc_id in query_docs:
                ground_truth_data.append([query_id, doc_id])

        ground_truth_df = pd.DataFrame(ground_truth_data, columns=['query_id', 'doc_id'])
        ground_truth_df.to_csv(ground_truth_file, sep='\t', index=False, header=False)
        print(f"Ground truth file created: {ground_truth_file}")

    def calculate_map1(self):
        """Calculates the Mean Average Precision (MAP)."""
        query_results = defaultdict(list)
        for _, row in self.related_output.iterrows():
            query_results[row['query_id']].append(row['doc_id'])

        map_scores = []
        for query_id in self.ground_truth['query_id'].unique():
            relevant_docs = self.ground_truth[self.ground_truth['query_id'] == query_id]['doc_id'].tolist()
            precisions = []
            num_relevant = 0
            for i, doc_id in enumerate(query_results[query_id]):
                if doc_id in relevant_docs:
                    num_relevant += 1
                    precisions.append(num_relevant / (i + 1))
            if precisions:
                map_scores.append(sum(precisions) / len(relevant_docs))

        return np.mean(map_scores)

    def calculate_map2(self):
        """Calculates the Mean Average Precision (MAP@1) for a given query."""
        total_ap = 0
        num_queries = len(self.related_output['query_id'].unique())

        for i, row in self.related_output.iterrows():
            relevant_documents = self.ground_truth[self.ground_truth['doc_id'] == row['doc_id']]
            if not relevant_documents.empty:
                precision_at_rank = 1 / (i + 1)
                total_ap += precision_at_rank

        if num_queries > 0:
            map1 = total_ap / num_queries
        else:
            map1 = 0

        print(f"MAP@1: {map1:.4f}")
        return map1

    def calculate_map(self):
        """Calculates the Mean Average Precision (MAP)."""
        query_results = defaultdict(list)
        for _, row in self.related_output.iterrows():
            query_results[row['query_id']].append(row['doc_id'])

        map_scores = []
        for query_id, relevant_docs in self.ground_truth.groupby('query_id')['doc_id'].apply(list).items():
            if query_id in query_results:
                docs = query_results[query_id]
                precisions = []
                num_relevant = 0
                for i, doc_id in enumerate(docs):
                    if doc_id in relevant_docs:
                        num_relevant += 1
                        precisions.append(num_relevant / (i + 1))
                if precisions:
                    ap = sum(precisions) / len(relevant_docs)
                    map_scores.append(ap)

        return np.mean(map_scores) if map_scores else np.nan

    def calculate_recall(self):
        """Calculates the Recall."""
        all_relevant_docs = set(self.ground_truth['doc_id'].tolist())
        retrieved_docs = set(self.related_output['doc_id'].tolist())
        return len(all_relevant_docs.intersection(retrieved_docs)) / len(all_relevant_docs)

    def calculate_precision_at_k(self, k=10):
        """Calculates the Precision at k."""
        top_k_docs = self.related_output['doc_id'].tolist()[:k]
        relevant_docs = set(self.ground_truth['doc_id'].tolist())
        return len(set(top_k_docs).intersection(relevant_docs)) / k

    def calculate_mrr(self):
        """Calculates the Mean Reciprocal Rank (MRR)."""
        mrr_scores = []
        for query_id in self.ground_truth['query_id'].unique():
            relevant_docs = self.ground_truth[self.ground_truth['query_id'] == query_id]['doc_id'].tolist()
            for i, doc_id in enumerate(self.related_output[self.related_output['query_id'] == query_id]['doc_id']):
                if doc_id in relevant_docs:
                    mrr_scores.append(1 / (i + 1))
                    break
        return np.mean(mrr_scores)

    def sort_results(self):
        """Sorts the results based on evaluation metrics."""
        # You can choose which metric to sort by here
        # Example: Sorting by MAP
        sorted_results = self.related_output.sort_values(by=['query_id', 'doc_id'])
        return sorted_results

    def evaluate(self):
        """Evaluates the search results and prints the metrics."""
        map_score = self.calculate_map()
        recall_score = self.calculate_recall()
        precision_at_10 = self.calculate_precision_at_k(k=10)
        mrr_score = self.calculate_mrr()
        return {
            "map_score": map_score,
            "recall_score": recall_score,
            "precision_at_10": precision_at_10,
            "mrr_score": mrr_score,
            "sorted_results": self.sort_results()
        }

# Example usage:
