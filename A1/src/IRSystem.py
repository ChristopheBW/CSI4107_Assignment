import math
import re  # regular expression to filter out punctuations
import multiprocessing
import Extractor


class IRSystem:

    def __init__(self, collection_path, query_path):
        self.extractor = Extractor.Extractor(collection_path, query_path)
        self.collection = self.extractor.get_collection()
        self.queries = self.extractor.get_queries()
        self.N = len(self.collection)
        self.inverted_index = {}
        self.load_tf_idf()

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

    # tf-idf implementation by "NullPointer"
    def load_tf_idf(self, file_path=None):
        '''Calculates the tf-idf weight for each term in each document with multiprocessing.
        O(N * M) where N is the number of documents and M is the avg number of terms

        :param file_path: the file path to load the inverted index
        :type file_path: str
        :return: the hashmap of tf-idf weight for each term in each document
        :rtype: dict
        '''

        if file_path is None:
            # tf_df: a hashmap of term frequency in each document and document frequency
            #   structure: term -> {docno -> tf, ... , "df" -> df}
            #   example: "the" -> {d0 -> 2, d1 -> 1, d2 -> 3, "df" -> 6}
            tf_df = {}

            for docno, tokens in self.collection.items():
                for token in tokens:
                    if token not in tf_df:
                        tf_df[token] = {"df": 0}
                    if docno not in tf_df[token]:
                        tf_df[token][docno] = 1
                    else:
                        tf_df[token][docno] += 1
                    tf_df[token]["df"] += 1

            print("tf_df done")
            # print('tf_df: ', tf_df)

            # tf-idf: a hashmap of tf-idf weight for each term in each document
            #   structure: term -> {docno -> tf-idf}
            #   example: "the" -> {d0 -> 0.1, d1 -> 0.05, d2 -> 0.15}
            self.inverted_index = {}
            for term in tf_df:
                self.inverted_index[term] = {}
                for docno in tf_df[term]:
                    if docno == "df":
                        continue
                    self.inverted_index[term][docno] = tf_df[term][docno] * math.log2(
                        self.N / tf_df[term]["df"])

            print("tf_idf done")
        else: # load from file
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    term, docnos = line.split(":")
                    self.inverted_index[term] = {}
                    for docno in docnos.split():
                        self.inverted_index[term][docno] = float(docnos.split()[1])

    def save_inverted_index(self, file_path):
        """ save the inverted index to a file

        :param file_path: the file path to save the inverted index
        :type file_path: str
        :return: None
        """

        with open(file_path, "w") as f:
            for term in self.inverted_index:
                f.write(term + ": ")
                for docno in self.inverted_index[term]:
                    f.write(docno + " " + str(self.inverted_index[term][docno]) + " ")
                f.write("\n")

    def get_cosine_similarity(self, query):
        '''Calculates the cosine similarity between the query and each document.

        :param query: the tokenized query
        :type query: list
        :return: ranked list of documented in reversed order of their relevance.
        :rtype: list
        '''

        # calculate the query vector
        print(query)
        query_vector = {}
        for term in query:
            if term not in query_vector:
                query_vector[term] = 1
            else:
                query_vector[term] += 1

        # calculate the document vector
        document_vectors = {}
        for term in query_vector:
            if term in self.inverted_index:
                for docno in self.inverted_index[term]:
                    if docno not in document_vectors:
                        document_vectors[docno] = {}
                    document_vectors[docno][term] = self.inverted_index[term][docno]

        # calculate the cosine similarity
        cosine_similarities = {}
        for docno in document_vectors:
            cosine_similarities[docno] = 0
            for term in query_vector:
                if term in document_vectors[docno]:
                    cosine_similarities[docno] += query_vector[term] * document_vectors[docno][term]

        # normalize the cosine similarity
        for docno in cosine_similarities:
            cosine_similarities[docno] /= math.sqrt(len(query_vector) * len(document_vectors[docno]))

        # sort the cosine similarity
        cosine_similarities = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)

        return cosine_similarities[:50]

if __name__ == '__main__':
    irs = IRSystem("./collection.txt", "./topics1-50.txt")
    # irs.save_inverted_index("./inverted_index.txt")
    ranked_list = irs.get_cosine_similarity(irs.queries["1"])
    print(ranked_list)
