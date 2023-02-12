import os  # used to explore the file system
import re  # used to search for the pattern
import multiprocessing  # used to parallelize the extraction
from nltk.stem import PorterStemmer  # find the root of the word


class Extractor:

    def __init__(self, collection_path, query_path):
        """ Initialize the Extractor class with the collection and query paths

        :param collection_path: path to the collection file
        :type collection_path: str
        :param query_path: path to the query file
        :type query_path: str
        """

        self.collection_path = collection_path
        self.query_path = query_path
        self.queries = {}
        self.load_query()
        self.collection = {}

        # if collection_path points to a file, load the collection from the file
        if os.path.isfile(collection_path):
            self.load_collection(collection_path)
        else:
            self.load_collection()

    def tokenize(self, text) -> list[str]:
        '''Tokenize the text

        :param text: the text to tokenize
        :type text: str
        :return: the tokenized text
        :rtype: list
        '''

        # def the pattern of the regex to filting
        regex_pattern = r'\w+'
        tokens_with_stopword = re.findall(regex_pattern,
                                          text)  # tokenize the text

        # print(tokens_with_stopword)

        # use the stopwords provided by the prof
        f = open('StopWords', 'r')
        stop_words = f.read().splitlines()

        # remove the stopwords
        tokens_cleaned = [
            w for w in tokens_with_stopword
            if w.lower() not in stop_words and w.isalpha()
        ]

        # stem word using PorterStemmer
        ps = PorterStemmer()
        tokens_stemmed = [ps.stem(w) for w in tokens_cleaned]

        return tokens_stemmed

    def parse_file(self, filename):
        """ parse the given file and extract the DOCNO and TEXT

        :param filename: the file to parse
        :type filename: str
        :return: a hashmap of the document number and inverted index
        :rtype: dict
        """
        # Old code to parse the file by using regex
        #     # Read the contents of the file
        #     buffer = f.read()
        #     # Extract doc tags from the file by using regex
        #     doc = self.extract_tag('DOC', buffer)
        #     # Extract docno and text tags from the file by using regex
        #     for d in doc:
        #         # !!! Assume there always has a space after <docno> and before </docno> !!!
        #         docno = self.extract_tag('DOCNO', d)[0]
        #         text = ' '.join(self.extract_tag('TEXT', d))
        #         # Tokenize the text
        #         tokens = Tokenizer.tokenize(text)
        #         # Add the docno and tokenized text to the hashmap
        #         doc_text_map[docno] = tokens

        print(filename)

        with open(os.path.join(self.collection_path, filename), 'r') as file:

            flag_doc = False  # indicate whether the current line is in a doc
            flag_text = False  # indicate whether the current line is in a text
            docno = ""
            text = ""
            docno_inversed_index_map = {}

            for line in file:
                # print(line)
                if "<DOC>" in line:
                    if flag_doc:
                        raise Exception("Error: <DOC> tag is not closed")
                    flag_doc = True
                if "<DOCNO>" in line:
                    if not flag_doc:
                        raise Exception("Error: <DOC> tag is not opened")
                    if flag_text:
                        raise Exception("Error: <TEXT> tag is not closed")
                    docno = re.findall(r'<DOCNO> (.*?) </DOCNO>', line)[0]
                if "</TEXT>" in line:
                    if not flag_doc:
                        raise Exception("Error: <DOC> tag is not opened")
                    flag_text = False
                elif "<TEXT>" in line:
                    if not flag_doc:
                        raise Exception("Error: <DOC> tag is not opened")
                    if flag_text:
                        raise Exception("Error: <TEXT> tag is not closed")
                    flag_text = True
                elif flag_text:
                    text += line
                if "</DOC>" in line:
                    if not flag_doc:
                        raise Exception("Error: </DOC> tag is not opened")
                    flag_doc = False
                    tokens = self.tokenize(text)
                    docno_inversed_index_map[docno] = tokens
                    docno = ""
                    text = ""

        return docno_inversed_index_map

    def load_collection(self, output_path=None):
        """ Load all files from the collection path, map the docno and tokens

        :return: a hashmap of the document number and tokens
        :rtype: dict
        """

        if output_path is not None:
            if not os.path.exists(output_path):
                raise Exception("Error: the output path does not exist")
            self.collection = self.load_collection_from_file(output_path)
        else:
            pool = multiprocessing.Pool()
            process_list = []

            # Iterate through the files in the collection path
            for filename in os.listdir(self.collection_path):
                # if filename != "AP880819":
                #     continue
                process_list.append(
                    pool.apply_async(self.parse_file, args=(filename, )))

            pool.close()
            pool.join()

            for process in process_list:
                self.collection.update(process.get())

    def load_query(self):
        """ Load the file from the query path, map the query number and tokens

        :return: a hashmap of the query number and tokens
        :rtype: dict
        """

        # Open the file and read the contents
        with open(self.query_path, 'r') as f:
            # Read the contents of the file
            query = re.findall(r'<top>(.*?)</top>', f.read(),
                               re.MULTILINE | re.DOTALL)

            for q in query:
                # Extract query number and title from the file by using regex
                queryno = re.findall(r'<num>(.*?)\n', q)
                title = re.findall(r'<title>(.*?)\n', q)
                # Tokenize the title
                tokens = self.tokenize(title[0])
                # Add the query number and tokenized title to the hashmap
                
                self.queries[queryno[0].strip()] = tokens

    def save_collection(self, output_path):
        """ Save the collection to the given path

        :param output_path: the path to save the collection
        :type output_path: str
        """

        with open(output_path, 'w') as f:
            for docno, tokens in self.collection.items():
                f.write(docno + "\t" + " ".join(tokens) + "\n")

    def load_collection_from_file(self, input_path):
        """ Load the collection from the given path

        :param input_path: the path to load the collection
        :type input_path: str
        :return: a hashmap of the document number and tokens
        :rtype: dict
        """

        collection = {}

        with open(input_path, 'r') as f:
            for line in f:
                docno, tokens = line.split("\t", 1)
                collection[docno] = tokens.split()

        return collection

    def get_collection(self):
        """ Get the collection

        :return: a hashmap of the document number and tokens
        :rtype: dict
        """

        return self.collection

    def get_queries(self):
        """ Get the query

        :return: a hashmap of the query number and tokens
        :rtype: dict
        """

        return self.queries


if __name__ == '__main__':
    extractor = Extractor('./Collection/', './topics1-50.txt')
    extractor.save_collection('./collection.txt')

    extractor = Extractor('./collection.txt', './topics1-50.txt')

    print(extractor.collection["AP881001-0003"])
