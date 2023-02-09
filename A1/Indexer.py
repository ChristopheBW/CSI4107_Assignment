import math
import Tokenizer # Path: ./Tokenizer.py
from collections import defaultdict

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


def getCosineSimilarity(query, tf_idf):

    '''Calculates the cosine similarity of each indexed documents as query words are processed one by one.

    :param query: the query in text
    :type query: list
    :param tf_idf
    :type tf_idf: dict
    :return: ranked list of documented in reversed order of their relevance.
    :rtype: list
    '''

    # Finds the number of documents that contain one or more query words.
    # The list then contains the documents that have the one of the query word.
    docContain = []
    for term in query:
        if term in tf_idf:
            for i in range(len(tf_idf[term])):
                if tf_idf[term][i] > 0:
                    docContain = list(set(docContain + [i]))


    # Calculates the cosine similarity of each indexed documents and the query words.
    # Incrementally computes cosine similarity of each indexed documents as query words are
    # processed one by one.
    cosSimilarity = {}
    
    for doc in docContain:
        cosSimilarity[doc] = 0
        querytf_idf = 0
        doctf_idf = 0
        for term in query:
            if term in tf_idf:
                querytf_idf += tf_idf[term][doc]
                doctf_idf += math.pow(tf_idf[term][doc], 2)

        sumQ = 0.0
        for term in tf_idf:
            sumQ += math.pow(tf_idf[term][doc],2)

        queryMagnitude = math.sqrt(sumQ)
        docMagnitude = math.sqrt(doctf_idf)
        
        # Result measurement should be values that are bound by a constrained range of 0 and 1.
        cosSimilarity[doc] = querytf_idf / (queryMagnitude * docMagnitude)
        
    print(cosSimilarity)
        
    # Sorts the result based on the value of cosine similarity.
    # Returns the documents in descending order of their relevance.
    # It is ordered in reverse order of their relevance.
    relevantDoc = list(cosSimilarity.items())
    relevantDoc.sort(key = lambda a: a[1], reverse = True)
    
    # List that is ranked contains the ids of the relevant document.
    rankedList = []
    for id in relevantDoc:
        rankedList.append(id[0])

    return rankedList


# TODO: Delete the following code before the submission
# The following code is for testing
if __name__ == '__main__':
    docs = [
        "Wasserman's control comes more from the respect he commands than from the stock he controls.",
        "Wasserman owns 6.9 percent of MCA's stock and controls another 8.9 percent through a series of trusts, including those of the family of MCA founder Jules Stein.",
        "From its beginning as a Hollywood talent-booking agency, MCA has grown into a powerhouse in producing movies, television programming, records and toys."
    ]
    tokenized_docs = [Tokenizer.tokenize(doc) for doc in docs]
    #for doc in docs:
        #print(doc)

    #print("\ntf-idf:\n", tf_idf(tokenized_docs))
    query = Tokenizer.tokenize("Wasserman's control comes more from the respect he commands than from the stock he controls.")
    print(getCosineSimilarity(query,tf_idf(tokenized_docs)))

    
            
