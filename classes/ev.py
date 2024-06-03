# input_file = 'D:/ir_final_final_final_the_flinalest/data/antiqe_output/output_collection.tsv'
# output_file = 'D:/ir_final_final_final_the_flinalest/data/antiqe_output/line_ids.txt'

# with open(input_file, 'r') as f, open(output_file, 'w') as out_f:
#     for line in f:
#         # Split the line based on the space delimiter
#         parts = line.strip().split(' ')  # Change delimiter here
#         if len(parts) == 2:
#             document_id, text = parts
#             out_f.write(document_id + '\n')

# def get_lines_from_indices(indices_file, original_file):
    # """
    # Reads indices from a file and returns the corresponding lines from an original file.

    # Args:
    #     indices_file: Path to the file containing indices (one index per line).
    #     original_file: Path to the original file containing lines.

    # Returns:
    #     A list of lines from the original file corresponding to the given indices.
    # """

# def get_line_numbers(dataset_name, query):
#     """Retrieves line numbers of documents relevant to the query.

#     Args:
#         dataset_name: The name of the dataset ('antique' or 'wikir').
#         query: The search query.

#     Returns:
#         A list of line numbers of related documents.
#     """

#     doc_file = 'D:/ir_final_final_final_the_flinalest/data/antiqe_output/output_collection.tsv'
#     matcher = Matching_Ranking(doc_file)
#     doc_vectors = matcher.data_rep.vsm
#     related_line_numbers = matcher.match(doc_vectors)
#     return related_line_numbers




# recall@k function
def recall(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = round(len(act_set & pred_set) / float(len(act_set)), 2)
    return result

# Read actual values from a file
with open('D:/ir_final_final_final_the_flinalest/data/antiqe_output/line_ids.txt', 'r') as f:
    actual = [line.strip() for line in f]  # Keep as strings

# Read predicted values from a file
with open('D:/ir_final_final_final_the_flinalest/data/related_doc_text.txt', 'r') as f:
    predicted = [line.strip() for line in f]  # Keep as strings

for k in range(1, 9):
    print(f"Recall@{k} = {recall(actual, predicted, k)}")



# def load_actual_data_mapping(data_file):
#     """Loads the actual data mapping from a file.

#     Args:
#         data_file: The path to the data file.

#     Returns:
#         A dictionary mapping document IDs to line numbers.
#     """

#     actual_data_mapping = {}
#     with open(data_file, 'r', encoding='utf-8') as f:
#         for line_number, line in enumerate(f):
#             doc_id, _ = line.strip().split('\t')  # Assuming tab as delimiter
#             actual_data_mapping[doc_id] = line_number

#     return actual_data_mapping

# # Load the mapping
# actual_data_mapping = load_actual_data_mapping('path/to/your/data_file.txt') 
