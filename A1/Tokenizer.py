import re  # regular expression to filter out punctuations
from nltk.stem import PorterStemmer  # find the root of the word


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

    # stem word using PorterStemmer
    ps = PorterStemmer()
    tokens_stemmed = [ps.stem(w) for w in tokens_cleaned]

    return tokens_stemmed


if __name__ == '__main__':
    text = "This search will be somewhat limiting in that it focuses on one particular medical condition among a broader group which are thought to be caused by the auto-immune system.  Efforts will be directed toward finding any reports of new ideas of cause as well as elaboration on the auto-immune theories.  The search will focus also on research efforts to find treatments to alleviate the symptoms of MS and/or to discover possible cures for it.  The names of companies involved and the particular drug or therapy should be a part of the relevant item.  Also relevant would be non-medical or alternative healing procedures which find acceptance among MS sufferers, e.g., bee venom, acupuncture, etc."

    tokens = getTokens(text)
    print(text, "\n\n", tokens)
