import math
import re  # regular expression to filter out punctuations


class IRSystem:

    # tf-idf implementation by "GEGEFE"
    def getInvertedIndex(self, tokens1):
        '''
        get the inverted index
        :param tokens: the tokenized text (list)
        :return: the inverted index (dict)
        '''

        tokens2 = [tokens1]
        inverted_index, counters = {}, {
        }  # use hashmap to store the inverted index

        # iter through the tokens
        for i, tokens in enumerate(tokens2):
            for token in tokens:

                if token not in inverted_index:
                    inverted_index[token] = [i]
                else:
                    inverted_index[token].append(i)
                if token not in counters:
                    counters[token] = 1
                else:
                    counters[token] += 1

        inverse_document_frequency = {}
        for word, document_indices in inverted_index.items():
            inverse_document_frequency[word] = math.log(
                len(tokens2) / len(document_indices))
        print(inverse_document_frequency)

        weights = {}
        for document_index, tokens in enumerate(tokens2):
            for word in tokens:
                term_frequency = tokens.count(word) / len(tokens)
                weights[(word, document_index
                         )] = term_frequency * inverse_document_frequency[word]
        print(weights)
        print(inverted_index)
        return inverted_index, weights

    # tf-idf implementation by "NullPointer"
    def tf_idf(self, tokenized_docs):
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

    def getCosineSimilarity(self, query, tf_idf):
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
                sumQ += math.pow(tf_idf[term][doc], 2)

            queryMagnitude = math.sqrt(sumQ)
            docMagnitude = math.sqrt(doctf_idf)

            # Result measurement should be values that are bound by a constrained range of 0 and 1.
            cosSimilarity[doc] = querytf_idf / (queryMagnitude * docMagnitude)

        print(cosSimilarity)

        # Sorts the result based on the value of cosine similarity.
        # Returns the documents in descending order of their relevance.
        # It is ordered in reverse order of their relevance.
        relevantDoc = list(cosSimilarity.items())
        relevantDoc.sort(key=lambda a: a[1], reverse=True)

        # List that is ranked contains the ids of the relevant document.
        rankedList = []
        for id in relevantDoc:
            rankedList.append(id[0])

        return rankedList
