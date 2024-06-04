import sys
import os

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
resource_path = os.path.join(current_dir, "..", "resources")
output_path = os.path.join(current_dir, "..", "outputs")

antique_input_file = os.path.join(resource_path, 'antique', 'antique-collection.txt')
antique_output_file = os.path.join(output_path, 'antique', 'output_collection.tsv')
antique_tfidf_file = os.path.join(output_path, 'antique', 'tfidf_results.pkl')
antique_related_file = os.path.join(output_path, 'antique', 'related.tsv')
antique_ground_truth_file = os.path.join(output_path, 'antique', 'ground_truth.tsv')

wikir_input_file = os.path.join(resource_path, 'wikir', 'wikir-collection.txt')
wikir_output_file = os.path.join(output_path, 'wikir', 'output_collection.tsv')
wikir_tfidf_file = os.path.join(output_path, 'wikir', 'tfidf_results.pkl')
wikir_related_file = os.path.join(output_path, 'wikir', 'related.tsv')
wikir_ground_truth_file = os.path.join(output_path, 'wikir', 'ground_truth.tsv')

query_output_file = os.path.join(output_path, 'query', 'output.tsv')
query_tfidf_file = os.path.join(output_path, 'query', 'tfidf.pkl')



def print_progress_bar(iteration, total, prefix='', suffix=''):
    """
    Prints a progress bar to the terminal.

    Args:
        iteration (int): Current iteration number.
        total (int): Total number of iterations.
        prefix (str, optional): Prefix to display before the bar. Defaults to ''.
        suffix (str, optional): Suffix to display after the bar. Defaults to ''.
    """
    filled_length = int(round(iteration / total * 50))  # Adjust bar length as needed
    bar = '█' * filled_length + '-' * (50 - filled_length)
    percent = round(iteration / total * 100, 1)
    sys.stdout.write(f'\r{prefix}|{bar}| {percent}% {suffix}')
    sys.stdout.flush()
