import re # regular expression to filter out punctuations
import math # math to do log calcualtions

def getTokens(text) -> list[str]:
    '''
    tokenize the text
    :param text: the text to tokenize (string)
    :return: the tokenized text (list)
    '''

    # def the pattern of the regex to filting out punctuations
    regex_pattern = r'\w+'
    tokens_with_stopword = re.findall(regex_pattern, text)  # tokenize the text

    # print(tokens_with_stopword)

    # use the stopwords provided by the prof
    f = open('StopWords', 'r')
    stop_words = f.read().splitlines()

    # remove the stopwords
    tokens_cleaned = [
        w for w in tokens_with_stopword if w.lower() not in stop_words]

    return tokens_cleaned


# TODO: this function is still in dev
def getInvertedIndex(tokens1) -> dict[str, list[int]]:
    '''
    get the inverted index
    :param tokens: the tokenized text (list)
    :return: the inverted index (dict)
    '''

    tokens2 = [tokens1]
    inverted_index,counters = {},{}  # use hashmap to store the inverted index
    
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
        inverse_document_frequency[word] = math.log(len(tokens2) / len(document_indices))
    print(inverse_document_frequency)


    weights = {}
    for document_index, tokens in enumerate(tokens2):
        for word in tokens:
            term_frequency = tokens.count(word) / len(tokens)
            weights[(word, document_index)] = term_frequency * inverse_document_frequency[word]
    print(weights)
    print(inverted_index)
    return inverted_index, weights


if __name__ == '__main__':
    text = "This search will be somewhat limiting in that it focuses on one particular medical condition among a broader group which are thought to be caused by the auto-immune system.  Efforts will be directed toward finding any reports of new ideas of cause as well as elaboration on the auto-immune theories.  The search will focus also on research efforts to find treatments to alleviate the symptoms of MS and/or to discover possible cures for it.  The names of companies involved and the particular drug or therapy should be a part of the relevant item.  Also relevant would be non-medical or alternative healing procedures which find acceptance among MS sufferers, e.g., bee venom, acupuncture, etc."

    tokens = getTokens(text)
    inverted_index = getInvertedIndex(tokens)

    print(tokens)