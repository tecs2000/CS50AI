import nltk
import sys
import os
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)

    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    mapping = dict()

    for file in os.listdir(directory):
        arquivo = open(os.path.join(directory, file))
        mapping[file] = arquivo.read()
        arquivo.close()

    return mapping


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    contents = []

    contents.extend([
        word.lower() for word in
        nltk.word_tokenize(document)
        if any(c.isalpha() for c in word)
    ])

    contents = [word for word in contents
                if word not in nltk.corpus.stopwords.words('english')]

    return contents


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    mapping = dict()

    # frequency calculation
    for file, words in documents.items():
        for word in set(words):
            mapping[word] = 1 if word not in mapping \
                else mapping[word] + 1

    # idf calculation
    for word in list(mapping):
        mapping[word] = math.log(len(documents) / mapping[word])

    return mapping


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    mapping = dict()

    for file, text in files.items():
        mapping[file] = 0
        for word in query:
            if word in list(idfs):
                mapping[file] += text.count(word) * idfs[word]
            else:
                mapping[file] += 0

    rank = sorted(mapping.items(), key=lambda x: x[1], reverse=True)
    return [element[0] for element in rank][:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    mapping = dict()

    for phrase in sentences:
        mapping[phrase] = [0, 0]

        for word in query:
            if word in sentences[phrase]:
                mapping[phrase][0] += idfs[word]
                mapping[phrase][1] += 1

        mapping[phrase][1] /= len(sentences[phrase])

    # caution with the density based choice!!!
    # its better when using only lambda x: x[1][0]
    rank = sorted(mapping.items(),
                  key=lambda x: (x[1][0], x[1][1]), reverse=True)

    return [element[0] for element in rank][:n]


if __name__ == "__main__":
    main()
