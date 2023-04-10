import math
import time
import multiprocessing
import Extractor
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing.pool import ThreadPool as Pool

def expand_query(query, model, top_n=5, similarity_threshold=0.7):
    '''
    Query Expansion method using 'fastText' model.
    input query, model, number of nearest neighbors, threshold
    returns expanded query
    '''
    expanded_query = set([query])  # use set instead of list to avoid duplicates
    tokens = query.split()
    
    # Get nearest neighbors for each token in the query
    for token in tokens:

        # Use a generator expression to avoid creating a temporary list
        neighbors = model.get_nearest_neighbors(token, k = top_n)
        important_neighbors = (neighbor for neighbor in neighbors if neighbor[0] > similarity_threshold)
        nearest_neighbors = list(important_neighbors)

        # Use list comprehension to add nearest neighbors to the expanded query
        expanded_query.update(neighbor[1] for neighbor in nearest_neighbors)
        
    #return the expanded query
    return list(expanded_query)
    
class IRSystem:

    def __init__(self, collection_path, query_path):
        self.extractor = Extractor.Extractor(collection_path, query_path)
        self.collection = self.extractor.get_collection()
        self.queries = self.extractor.get_queries()
        self.N = len(self.collection)
        self.Q = len(self.queries)

        self.inverted_index = {}
        self.model = fasttext.load_model("./model.bin")
        # self.load_tf_idf()

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
        

        weights = {}
        for document_index, tokens in enumerate(tokens2):
            for word in tokens:
                term_frequency = tokens.count(word) / len(tokens)
                weights[(word, document_index
                         )] = term_frequency * inverse_document_frequency[word]
        
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
            self.inverted_index = {}

            for docno, tokens in self.collection.items():
                for token in tokens:
                    if token not in self.inverted_index:
                        self.inverted_index[token] = {"df": 0}
                    if docno not in self.inverted_index[token]:
                        self.inverted_index[token][docno] = 1
                    else:
                        self.inverted_index[token][docno] += 1
                    self.inverted_index[token]["df"] += 1

            print("tf_df done")
            # print('tf_df: ', tf_df)

            # tf-idf: a hashmap of tf-idf weight for each term in each document
            #   structure: term -> {docno -> tf-idf}
            #   example: "the" -> {d0 -> 0.1, d1 -> 0.05, d2 -> 0.15}
            for term in self.inverted_index:
                term_map = self.inverted_index[term]
                self.inverted_index[term] = {}
                for docno in term_map:
                    if docno != "df":
                        self.inverted_index[term][docno] = term_map[
                            docno] * math.log2(self.N / term_map["df"])

            print("tf_idf done")
        else:  # load from file
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    term, docnos = line.split(":")
                    self.inverted_index[term] = {}
                    for docno in docnos.split():
                        self.inverted_index[term][docno] = float(
                            docnos.split()[1])

    def sklearn_cosine_similarity(self, query, i):
        """ calculate the cosine similarity between the query and each document with sklearn

        :param query: the query
        :type query: str
        :param i: the index of the query
        :type i: int
        :return: the cosine similarity between the query and each document
        :rtype: list
        """

        vectorizer = TfidfVectorizer(stop_words="english")
        doc_query_text = list(self.collection.values()) + [query]
        # print(doc_query_text[-1:])
        tfidf = vectorizer.fit_transform(doc_query_text)

        doc_tf_idf = tfidf[:len(self.collection)]
        query_tf_idf = tfidf[len(self.collection):]

        cosine_sim_matrix = cosine_similarity(query_tf_idf[0], doc_tf_idf)
        # Get the cosine similarity for query 1 and all documents
        query_cosine_sims = cosine_sim_matrix[0]

        # Create a hashmap from "docno" to "cosine_sim" for query 1
        ranked_list = {
            docno: cosine_sim
            for docno, cosine_sim in zip(self.collection.keys(),
                                         query_cosine_sims)
        }

        sorted_dict = sorted(ranked_list.items(),
                                    key=lambda x: x[1],
                                    reverse=True)

        rank = 1
        result = ""
        for docno, score in sorted_dict:
            if rank <= 1000:
                result += f"{i}\tQ0\t{docno}\t{rank}\t{score}\tExp\n"
                # result += str(i) + "Q0 " + docno + " " + str(rank) + " " + str(score) + " Exp\n"
                rank += 1
        print(f"Query {i} done")

        return result


    def get_inverted_index(self):
        """ get the inverted index

        :return: the inverted index
        :rtype: dict
        """

        return self.inverted_index

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
                    f.write(docno + " " +
                            str(self.inverted_index[term][docno]) + " ")
                f.write("\n")

    def get_cosine_similarity(self, query):
        '''Calculates the cosine similarity between the query and each document.

        :param query: the tokenized query
        :type query: set
        :return: ranked list of documented in reversed order of their relevance.
        :rtype: list
        '''
        ''''''
        # print(query)

        # calculate the query vector
        #   structure: term -> tf
        #   example: "the" -> 2
        query_vector = {}
        for term in query:
            if term not in query_vector:
                query_vector[term] = 1
            else:
                query_vector[term] += 1

        # calculate the document vector
        #   structure: docno -> {term -> tf-idf}
        #   example: d0 -> {"the" -> 0.1, "is" -> 0.05}
        document_vectors = {}
        for term in query_vector:
            if term in self.inverted_index:
                for docno in self.inverted_index[term]:
                    if docno not in document_vectors:
                        document_vectors[docno] = {}
                    document_vectors[docno][term] = self.inverted_index[term][
                        docno]

        # calculate the cosine similarity
        cosine_similarities = {}
        for docno in document_vectors:
            cosine_similarities[docno] = 0
            for term in query_vector:
                if term in document_vectors[docno]:
                    cosine_similarities[docno] += query_vector[
                        term] * document_vectors[docno][term]

        # normalize the cosine similarity
        '''
        for docno in cosine_similarities:
            cosine_similarities[docno] /= (math.sqrt(
                len(query_vector) * len(document_vectors[docno])))
        '''
        sum1 = 0
        sum2 = 0
        for docno in document_vectors:
            for term in query_vector:
                if term in document_vectors[docno]:
                    sum1 += math.pow(query_vector[term], 2)
                    sum2 += math.pow(document_vectors[docno][term], 2)

            sum1 = math.sqrt(sum1)
            sum2 = math.sqrt(sum2)
            cosine_similarities[docno] /= (sum1 * sum2)

        # sort the cosine similarity
        cosine_similarities = sorted(cosine_similarities.items(),
                                     key=lambda x: x[1],
                                     reverse=True)

        return cosine_similarities

    def calculate_result(self):
        ''' Calculate the result of the queries with multiprocessing
        '''

        # multiprocessing
        pool = Pool()
        process_list = []

        self.stats = 0
        for i in range(1, self.Q + 1):
            expanded_query = " ".join(expand_query(self.queries[str(i)], self.model))
            print(self.queries[str(i)] + " sdwdwd")
            print(expanded_query)
            process_list.append(
                pool.apply_async(self.sklearn_cosine_similarity,
                                (expanded_query, i, ),))
        pool.close()
        pool.join()

        # write to file
        with open("results.txt", "w") as f:
            for i in range(len(process_list)):
                f.write(process_list[i].get())

if __name__ == '__main__':
    irs = IRSystem("./Collection", "./topics1-50.txt")
    print("Calculating...")
    irs.calculate_result()
    # irs.save_inverted_index("./inverted_index.txt")
    #print(irs.queries)
    #print(irs.queries["1"])
    # result = ""
    # rank = 1
    # with open('results.txt', 'w') as f:
    #     for i in range(1, len(irs.queries) + 1):
    #         # ranked_list = irs.get_cosine_similarity(irs.queries[str(i)])
    #         ranked_list = irs.sklearn_cosine_similarity(irs.queries[str(i)])
    #         sorted_dict = sorted(ranked_list.items(),
    #                                     key=lambda x: x[1],
    #                                     reverse=True)
    #         for docno, score in sorted_dict:
    #             result += str(i) + " Q0 " + docno + " " + str(rank) + " " + str(score) + " Exp\n"
    #             rank += 1
    #         rank = 1
    #     f.write(result)
    #

            # for j in range(len(ranked_list)):
            #     if j == 1001:
            #         break
            #     else:
            #         row = "{}\tQ0\t{}\t{}\t{}\tExp\n".format(
            #             i, ranked_list[j][0], rank, ranked_list[j][1])
            #         # row = str(i) + " Q0 " + str(
            #         #     ranked_list[j][0]) + " " + str(rank) + " " + str(
            #         #         ranked_list[j][1]) + " run_name" + "\n"
            #         rank += 1
            #         f.write(row)
    # ranked_list = irs.get_cosine_similarity(irs.queries["1"])
    # print(ranked_list[:10])
