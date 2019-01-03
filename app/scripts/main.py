import urllib
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import bs4
#import pyLDAvis as pyLDAvis
#import pyLDAvis.gensim
import spacy
from spacy.lang.en import English
import gensim
from gensim import corpora
import warnings
import pickle
from elasticsearch import Elasticsearch
from datetime import datetime

spacy.load('en')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# insert URL to scan here
#url = "https://www.heise.de/newsticker/meldung/Jugendliche-lieben-Netflix-und-WhatsApp-keiner-mag-Facebook-4234532.html"
#url = "https://www.theguardian.com/us-news/2018/dec/11/trump-meeting-pelosi-schumer-democrats-wall-border-funding-clash-debate-"
url = "https://en.wikipedia.org/wiki/Lemmatization"
#url = "https://steamcommunity.com/"


'''
##small bs test
page = requests.get(url)
soup = bs4.BeautifulSoup(page.content, 'html.parser')

##do this to see the website's html.
print(soup.prettify())

##do this to get all of the text in <p> tags
a = soup.findAll('p')
for b in a:
    print(b.getText())
'''


page = urllib.request.urlopen(url).read().decode('utf-8')
soup = bs4.BeautifulSoup(page, features='html.parser').getText()

tokens = nltk.word_tokenize(soup)
parser = English()


def tokenizeText(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def prepare_text_for_lda(text):
    tokens = tokenizeText(text)
    tokens = [token for token in tokens if len(token) > 4]
    #tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


collection = []
for line in tokens:
    #print(type(line))
    x = prepare_text_for_lda(line)
    if x:
        collection.append(x)

#print(collection)


dictionary = corpora.Dictionary(collection)
corpus = [dictionary.doc2bow(text) for text in collection]

pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

NUM_TOPICS = 1 #5 -> number of rows
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')

topics = ldamodel.print_topics(num_words=1000) #4 -> number of words
#print(topics)


# create a dict with key/value pairs from results
dictOfWords = {}

for topic in topics:
    topicList = topic[1].replace("\"", "").strip().split('+')
    for entry in topicList:
        entryList = entry.strip().split('*')
        #print(entryList[1] + " -> " + entryList[0])
        dictOfWords[entryList[1]] = float(entryList[0])
        #dictOfWords = {[entryList[1]]: entryList[0]}


'''
# for testing purposes only: count all hits and check total percentage
sumOfValues = 0.0
dictIndex = 1
for key, value in dictOfWords.items():
    value *= 100
    print(str(dictIndex) + ". -> " + "key: " + key + " | value: " + str(value))
    sumOfValues += value
    dictIndex += 1
#print("total percent: " + str(sumOfValues))
'''

# elasticsearch initiation and pushing dict keys into it
es = Elasticsearch()
esIndexName = "index_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
i = 1
wordSum = 0

for key, value in dictOfWords.items():
    value1k = int(value * 1000)
    wordSum += value1k
    print("total sum: " + str(wordSum) + " | word: " + key + " | value: " + str(value1k))
    for _ in range(value1k):
        doc = {'word': key}
        res = es.index(index=esIndexName, doc_type='data', id=i, body=doc)
        #print(res['result'])
        i += 1


'''
# stupid idea: add "Other" manually n times (= 1000-wordSum)
for _ in range(1000-wordSum):
    i += 1
    doc = {'word': "Other"}
    res = es.index(index=esIndexName, doc_type='data', id=i, body=doc)
'''

'''
# the old way: push key/value pair with value information
for key, value in dictOfWords.items():
    print(key + ": " + str(value) + " | id: " + str(i))
    doc = {'word': key, 'value': value}
    res = es.index(index=esIndexName, doc_type='data', id=i, body=doc)
    print(res['result'])
    i += 1
'''


'''
# refresh index and see result...not necessary, we use kibana :D
es.indices.refresh(index=esIndexName)

res = es.get(index=esIndexName, doc_type='data', id=1)
print(res['_source'])
'''


'''
# Create Dictionary
id2word = corpora.Dictionary(collection)

# Create Corpus
texts = collection

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

#print(lda_model.print_topics())
doc_lda = lda_model[corpus]

print('\nPerplexity: ', lda_model.log_perplexity(corpus))

dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = nltk.pickle.load(open('corpus.pkl', 'rb'))

lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)
'''
