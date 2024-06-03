import re
import string
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
# from gensim.models import Word2Vec
# from scipy import triu
import pandas as pd
import os

from utils import print_progress_bar, antique_input_file, antique_output_file


# import nltk
# nltk.download("stopwords")
# nltk.download('wordnet')

class TextProcessing:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, text):
        # Tokenize the given text into individual words.
        # Args:
        # text (str): The text to be tokenized.
        # Returns:
        # list: A list of tokens (words).
        # Example:
        # Input: "This is a sample text."
        # Output: ['This', 'is', 'a', 'sample', 'text.']
        return text.split()

    def remove_stop_words(self, tokens):
        # Remove stop words from the given list of tokens.
        # Args:
        # tokens (list): A list of tokens (words).
        # Returns:
        # list: A list of tokens with stop words removed.
        # Example:
        # Input: ['This', 'is', 'a', 'sample', 'text.']
        # Output: ['sample', 'text.']
        return [token for token in tokens if token.lower() not in self.stop_words]

    def stem_or_lemmatize(self, tokens, method='stem'):
        # Perform stemming or lemmatization on the given list of tokens.
        # Args:
        # tokens (list): A list of tokens (words).
        # method (str): The processing method to use, either 'stem' or 'lemmatize'.
        # Returns:
        # list: A list of processed tokens.
        # Example:
        # Input: ['running', 'ran', 'run']
        # Output (stemming): ['run', 'run', 'run']
        # Output (lemmatization): ['running', 'run', 'run']
        if method == 'stem':
            processed_tokens = [self.stemmer.stem(token) for token in tokens]
        elif method == 'lemmatize':
            processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        else:
            raise ValueError("Invalid method. Choose 'stem' or 'lemmatize'.")
        return processed_tokens

    def normalize_text(self, text):
        # Normalize the given text by performing various transformations.
        # Args:
        # text (str): The text to be normalized.
        # Returns:
        # str: The normalized text.
        # Example:
        # Input: "This is a  sample text.  "
        # Output: "this is a sample text"
        normalized_text = text.lower().strip()
        normalized_text = re.sub(r'[^a-zA-Z0-9\s]', '', normalized_text)  # Remove non-alphanumeric characters
        return normalized_text

    def remove_punctuation(self, text):
        # Remove punctuation from the given text.
        # Args:
        # text (str): The text to be processed.
        # Returns:
        # str: The text with punctuation removed.
        # Example:
        # Input: "This is a sample text!"
        # Output: "This is a sample text"
        return text.translate(str.maketrans('', '', string.punctuation))

    def sentence_tokenize(self, text):
        # Tokenize the given text into sentences.
        # Args:
        # text (str): The text to be tokenized.
        # Returns:
        # list: A list of sentences.
        # Example:
        # Input: "This is the first sentence. This is the second sentence."
        # Output: ['This is the first sentence.', 'This is the second sentence.']
        return sent_tokenize(text)

    def word_tokenize(self, text):
        # Tokenize the given text into words.
        # Args:
        # text (str): The text to be tokenized.
        # Returns:
        # list: A list of words.
        # Example:
        # Input: "This is a sample text."
        # Output: ['This', 'is', 'a', 'sample', 'text', '.']
        return word_tokenize(text)

    def pos_tag_tokens(self, tokens):
        # Perform Part-of-Speech tagging on the given tokens.
        # Args:
        # tokens (list): A list of tokens (words).
        # Returns:
        # list: A list of tuples, where each tuple is (token, POS tag).
        # Example:
        # Input: ['This', 'is', 'a', 'sample', 'text.']
        # Output: [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sample', 'NN'), ('text.', 'NN')]
        return pos_tag(tokens)

    def chunk_text(self, tokens):
        # Perform chunking on the given tokens.
        # Args:
        # tokens (list): A list of tokens (words).
        # Returns:
        # str: The chunked text in IOB format.
        # Example:
        # Input: ['This', 'is', 'a', 'sample', 'text.']
        # Output:
        # This B-NP
        # is B-VP
        # a B-NP
        # sample B-NP
        # text. B-NP
        tagged_tokens = self.pos_tag_tokens(tokens)
        iob_tagged_tokens = tree2conlltags(conlltags2tree(tagged_tokens))
        return "\n".join([" ".join([token, tag]) for token, tag in iob_tagged_tokens])

    def lemmatize_with_pos(self, tokens):
        # Lemmatize tokens using POS tagging for better accuracy.
        # Args:
        # tokens (list): A list of tokens (words).
        # Returns:
        # list: A list of lemmatized tokens.
        # Example:
        # Input: ['running', 'ran', 'run']
        # Output: ['running', 'run', 'run']
        tagged_tokens = self.pos_tag_tokens(tokens)
        lemmatized_tokens = []
        for token, tag in tagged_tokens:
            if tag.startswith('N'):  # Noun
                lemmatized_tokens.append(self.lemmatizer.lemmatize(token, pos='n'))
            elif tag.startswith('V'):  # Verb
                lemmatized_tokens.append(self.lemmatizer.lemmatize(token, pos='v'))
            elif tag.startswith('J'):  # Adjective
                lemmatized_tokens.append(self.lemmatizer.lemmatize(token, pos='a'))
            elif tag.startswith('R'):  # Adverb
                lemmatized_tokens.append(self.lemmatizer.lemmatize(token, pos='r'))
            else:
                lemmatized_tokens.append(self.lemmatizer.lemmatize(token))
        return lemmatized_tokens

    def tfidf_vectorize(self, texts):
        # Calculate TF-IDF vectors for a list of texts.
        # Args:
        # texts (list): A list of strings representing the texts.
        # Returns:
        # scipy.sparse.csr_matrix: A sparse matrix of TF-IDF vectors.
        # Example:
        # Input: ['This is a sample text.', 'Another sample text.']
        # Output: A sparse matrix with TF-IDF values for each word in the corpus.
        vectorizer = TfidfVectorizer()
        tfidf_vectors = vectorizer.fit_transform(texts)
        return tfidf_vectors

    # def train_word2vec(self, sentences, size=100, window=5, min_count=5):
    #     # Train a Word2Vec model on a list of sentences.
    #     # Args:
    #         # sentences (list): A list of lists, where each inner list represents a sentence.
    #         # size (int): The dimensionality of the word vectors.
    #         # window (int): The maximum distance between the current and predicted word within a sentence.
    #         # min_count (int): The minimum count of words to be included in the vocabulary.
    #     # Returns:
    #         # gensim.models.Word2Vec: A trained Word2Vec model.
    #     model = Word2Vec(sentences, size=size, window=window, min_count=min_count)
    #     return model

    def get_word_vector(self, word, model):
        # Get the vector representation of a word from a trained Word2Vec model.
        # Args:
        # word (str): The word to get the vector for.
        # model (gensim.models.Word2Vec): The trained Word2Vec model.
        # Returns:
        # numpy.ndarray: The vector representation of the word.
        return model.wv[word]

    def process_text(self, text, processing_method='lemmatize'):
        text = self.normalize_text(text)
        text = self.remove_punctuation(text)

        # Tokenize, remove stop words, and process (stem or lemmatize)
        tokens = self.tokenize(text)
        tokens = self.remove_stop_words(tokens)
        processed_tokens = self.stem_or_lemmatize(tokens, method=processing_method)
        if len(processed_tokens) == 1:
            return None
        results = " ".join(processed_tokens).replace(' ', '\t', 1)
        # Create a DataFrame for output
        df = pd.DataFrame({'tokens': [results]})
        return df

    def process_text_file(self, inputFile, outputFile, processing_method='lemmatize'):
        # Processes a text file, applies text processing methods, and saves the output to a TSV file.
        # Args:
        #     input_file (str): Path to the input text file.
        #     output_file (str): Path to the output TSV file.
        #     processing_method (str, optional): The processing method to use, either 'stem' or 'lemmatize'. Defaults to 'stem'.

        with open(inputFile, 'r', encoding="utf-8") as f:
            i = 0
            lines = f.readlines()
            total_lines = len(lines)
            for line in lines:
                i += 1
                print_progress_bar(i, total_lines, prefix='Processing', suffix='Complete')
                # Split the line based on the tab delimiter
                # parts = line.strip().split(' ')
                # if len(parts) == 2:  # Ensure there are two parts (ID and text)
                #     document_id, text = parts
                df = self.process_text(line, processing_method=processing_method)
                # Save to TSV file (append mode)
                if df is not None:
                    df.to_csv(outputFile, sep='\\', index=False, mode='a', header=False)


# processor = TextProcessing()
# processor.process_text_file(antique_input_file, antique_output_file, processing_method='lemmatize')
