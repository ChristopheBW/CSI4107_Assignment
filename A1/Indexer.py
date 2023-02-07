import math
import Tokenizer # Path: ./Tokenizer.py


def tf_idf(tokenized_docs):
    '''Calculates the tf-idf weight for each term in each document

    :param tokenized_docs: the tokenized documents
    :type tokenized_docs: list
    :return: the hashmap of tf-idf weight for each term in each document
    :rtype: dict
    '''

    # N: total number of documents
    N = len(tokenized_docs)

    # tf: a hashmap of term frequency in each document
    #   structure: term -> [tf_doc1, tf_doc2, ...]
    #   where tf_doc1 is the term frequency in document 1
    #   example: {'computer': [1, 0, 2, ..., 4], 'data': [2, 1, 3, ..., 0]}
    tf = {}
    for j in range(len(tokenized_docs)):
        doc = tokenized_docs[j]
        for term in doc:
            if term not in tf:
                tf[term] = [0] * N
            tf[term][j] += 1

    # print('tf: ', tf)

    # df: a hashmap of document frequency of each term
    #   structure: term -> df
    #   example: {'computer': 3, 'data': 5}
    df = {}
    for term in tf:
        df[term] = sum([1 for i in tf[term] if i > 0])

    # print('df: ', df)

    # idf: a hashmap of inverse document frequency of each term
    #   structure: term -> idf
    #   example: {'computer': 0.5, 'data': 0.2}
    idf = {}
    for term in df:
        idf[term] = math.log2(N / df[term])

    # print('idf: ', idf)

    # tf-idf: a hashmap of tf-idf weight for each term in each document
    #   structure: term -> [tf-idf_doc1, tf-idf_doc2, ...]
    #   where tf-idf_doc1 is the tf-idf weight of the term in document 1
    #   example: {'computer': [0.5, 0, 1, ..., 2], 'data': [1, 0.5, 1.5, ..., 0]}
    tf_idf = {}
    for term in tf:
        tf_idf[term] = [tf[term][i] * idf[term] for i in range(N)]

    return tf_idf


# TODO: Delete the following code before the submission
# The following code is for testing
if __name__ == '__main__':
    docs = [
        "Wasserman's control comes more from the respect he commands than from the stock he controls.",
        "Wasserman owns 6.9 percent of MCA's stock and controls another 8.9 percent through a series of trusts, including those of the family of MCA founder Jules Stein.",
        "From its beginning as a Hollywood talent-booking agency, MCA has grown into a powerhouse in producing movies, television programming, records and toys."
    ]
    tokenized_docs = [Tokenizer.tokenize(doc) for doc in docs]
    for doc in docs:
        print(doc)
    print("\nTokenized docs:\n", tokenized_docs)

    print("\ntf-idf:\n", tf_idf(tokenized_docs))

