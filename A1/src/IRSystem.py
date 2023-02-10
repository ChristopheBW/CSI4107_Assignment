import math
import re  # regular expression to filter out punctuations
import multiprocessing
import Extractor


class IRSystem:

    def __init__(self, collection_path, query_path):
        self.extractor = Extractor.Extractor(collection_path, query_path)
        self.collection = self.extractor.get_collection()
        self.queries = self.extractor.get_queries()
        self.docno_list = list(self.collection.keys())
        self.tokens_list = list(self.collection.values())
        self.N = len(self.docno_list)
        self.inverted_index = self.tf_idf(self.tokens_list)
        self.weights = {}

    # def get_inverted_index(self):
    #     """ get the inverted index
    #
    #     :return: the inverted index
    #     :rtype: dict
    #     """
    #
    #     return self.inverted_index

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

    def get_tf_df(self, doc, j):
        """ get the tf of each term in each document and df

        :param doc: the tokenized doc
        :type tokens: list
        :param j: the index of the doc
        :type j: int
        :return: the term frequency of each term in each document
        :rtype: dict
        """
        
        partial_tf_df = {}
        print("child_process: {}/{}".format(j, self.N))

        for term in doc:
            if term not in partial_tf_df:
                partial_tf_df[term] = [0] * (self.N + 1)
            partial_tf_df[term][j] += 1
            partial_tf_df[term][-1] += 1
        
        return partial_tf_df


    # tf-idf implementation by "NullPointer"
    def tf_idf(self, tokenized_docs):
        '''Calculates the tf-idf weight for each term in each document with multiprocessing.

        :param tokenized_docs: the tokenized documents
        :type tokenized_docs: list
        :return: the hashmap of tf-idf weight for each term in each document
        :rtype: dict
        '''

        # tf: a hashmap of term frequency in each document
        #   structure: term -> [tf_doc1, tf_doc2, ..., df]
        #   where tf_doc1 is the term frequency in document 1
        #   and the last element stores the total term frequency in all documents
        #   example: {'computer': [1, 0, 2, ..., 4], 'data': [2, 1, 3, ..., 8]}
        tf_df = {}
        pool = multiprocessing.Pool(16)
        process_list = []

        for j in range(len(tokenized_docs)):
            print("async: {}/{}".format(j, self.N))
            doc = tokenized_docs[j]
            # process_list.append(pool.apply_async(self.get_tf_df, (doc, j)))
            for term in doc:
                if term not in tf_df:
                    tf_df[term] = [0] * (self.N + 1)
                tf_df[term][j] += 1
                tf_df[term][-1] += 1

        pool.close()
        pool.join()

        # for process in process_list:
        #     partial_tf_df = process.get()
        #     for term, tf_df_list in partial_tf_df.items():
        #         if term not in tf_df:
        #             tf_df[term] = tf_df_list
        #         else:
        #             for i in range(len(tf_df_list)):
        #                 tf_df[term][i] += tf_df_list[i]

        print("tf_df done")
        # print('tf_df: ', tf_df)

        # tf-idf: a hashmap of tf-idf weight for each term in each document
        #   structure: term -> [tf-idf_doc1, tf-idf_doc2, ...]
        #   where tf-idf_doc1 is the tf-idf weight of the term in document 1
        #   example: {'computer': [0.5, 0, 1, ..., 2], 'data': [1, 0.5, 1.5, ..., 0]}
        tf_idf = {}
        for term in tf_df:
            tf_idf[term] = [tf_df[term][i] * math.log2(self.N / tf_df[term][-1]) for i in range(self.N)]

        return tf_idf

    def get_cosine_similarity(self, query, tf_idf):
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


if __name__ == '__main__':
    irs = IRSystem("./collection.txt", "./topics1-50.txt")
