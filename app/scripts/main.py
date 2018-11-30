import urllib
import nltk
import bs4
import pyLDAvis as pyLDAvis
import spacy
spacy.load('en')
from spacy.lang.en import English
import gensim

import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)


##dis the url u gon need
url = "https://www.heise.de/newsticker/meldung/Jugendliche-lieben-Netflix-und-WhatsApp-keiner-mag-Facebook-4234532.html"

'''
    ##small bs test
    page = requests.get(url)
    soup = bs4.BeautifulSoup(page.content, 'html.parser')

    ##do this to see the sites html.
    print(soup.prettify())


    d##o this to get all of the text in <p> tags
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


from nltk.corpus import wordnet as wn


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


from nltk.stem.wordnet import WordNetLemmatizer


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
 #   print(type(line))
    x = prepare_text_for_lda(line)
    if x :
     collection.append(x)

#print(collection)


from gensim import corpora
dictionary = corpora.Dictionary(collection )
corpus = [dictionary.doc2bow(text) for text in collection]

import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')


import gensim
NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')

topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)




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

import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)
'''