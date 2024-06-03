import numpy as np
import pandas as pd
from collections import defaultdict

from classes.utils import related_output_file, antique_output_file


class Evaluation:
    def __init__(self, related_output_file, ground_truth_file):
        """
        Initializes the Evaluation class.

        Args:
            related_output_file (str): Path to the related output TSV file.
            ground_truth_file (str): Path to the ground truth TSV file.
        """
        self.related_output = pd.read_csv(related_output_file, sep='\t', header=None, names=['doc_id', 'text'])
        self.ground_truth = pd.read_csv(ground_truth_file, sep='\t', header=None, names=['query_id', 'doc_id'])

    def calculate_map(self):
        """Calculates the Mean Average Precision (MAP)."""
        query_results = defaultdict(list)
        for _, row in self.related_output.iterrows():
            query_results[row['doc_id']].append(row['text'])

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
            for i, doc_id in enumerate(self.related_output['doc_id'].tolist()):
                if doc_id in relevant_docs:
                    mrr_scores.append(1 / (i + 1))
                    break
        return np.mean(mrr_scores)

    def sort_results(self):
        """Sorts the results based on evaluation metrics."""
        # You can choose which metric to sort by here
        # Example: Sorting by MAP
        sorted_results = self.related_output.sort_values(by=['doc_id'], ascending=False)
        return sorted_results

    def evaluate(self):
        """Evaluates the search results and prints the metrics."""
        map_score = self.calculate_map()
        recall_score = self.calculate_recall()
        precision_at_10 = self.calculate_precision_at_k(k=10)
        mrr_score = self.calculate_mrr()

        print(f"MAP: {map_score}")
        print(f"Recall: {recall_score}")
        print(f"Precision@10: {precision_at_10}")
        print(f"MRR: {mrr_score}")

        sorted_results = self.sort_results()
        print("\nSorted Results (by MAP):")
        print(sorted_results)


# Example usage:
evaluation = Evaluation(related_output_file, antique_output_file)
evaluation.evaluate()
