#import packages
#from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import gensim
from gensim import models
import matplotlib.pyplot as plt
import pyLDAvis.gensim

nltk.download('wordnet')

class TopicAnalysis(object):
    def __init__(self,speech):
        self.speech = speech


    def preprocess(self,stopwords):
        lemmatizer = WordNetLemmatizer()
        processed = []
        for s in self.speech:
            text = []
            for word in simple_preprocess(s):
                if word not in stopwords and len(word) > 2:
                    text.append(lemmatizer.lemmatize(word))
            processed.append(text)
        
        return processed

    # Vectorize texts by Bag-of-words
    def BoW_vectorizing(self,corpus):
        dictionary = Dictionary(corpus)
        dictionary.filter_extremes(no_below=2, no_above=0.8)
        bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]

        return dictionary, bow_corpus
    
    # Vectorize texts by TF-IDF
    def Tfidf_Vectorizing(self,bow_corpus):
        tfidf = models.TfidfModel(bow_corpus)
        tfidf_corpus = tfidf[bow_corpus]

        return  tfidf_corpus
        
    # Train LDA model with Bow model
    def LDA_by_BoW(self,n_topics,corpus):
        #text vectorizing
        dictionary, bow_corpus = self.BoW_vectorizing(corpus)
        
        # fitting model
        lda =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = n_topics, 
                                   id2word = dictionary,                                    
                                   passes = 10,
                                   workers = 2,random_state=2022)

        #get the topic words of each topic
        topic_words = self.get_topic_words(lda,n_topics)
        #get the dominant topic label of each speech
        labels = self.get_topic_label(lda,bow_corpus)

        return lda, topic_words, labels

    # Train LDA model with TFIDF
    def LDA_by_Tfidf(self,n_topics,corpus):
        # text vectorizing
        dictionary, bow_corpus = self.BoW_vectorizing(corpus)
        tfidf_corpus = self.Tfidf_Vectorizing(bow_corpus)

        # fitting model
        lda =  gensim.models.LdaMulticore(tfidf_corpus, 
                                   num_topics = n_topics, 
                                   id2word = dictionary,                                    
                                   passes = 10,
                                   workers = 2,random_state=2022)

        
        #get the topic words of each topic
        topic_words = self.get_topic_words(lda,n_topics)
        #get the dominant topic label of each speech
        labels = self.get_topic_label(lda,tfidf_corpus)

        return lda, topic_words, labels
    
    # To get the top words for each topic
    def get_topic_words(self,lda,n_topics,n_words=12):
        topic_words = []
        for i in range(n_topics):
            words = lda.show_topic(i,topn=n_words)
            topic_words.append([word[0] for word in words])
            
        return topic_words

    # To get the topic label for each speech
    def get_topic_label(self,lda,corpus):
        labels =[]
        for row in lda[corpus]:
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            labels.append(row[0][0]+1)
        return labels

    # Hyperparameter tuning: To find the most suitable numbers of topics
    def find_best_topic_numbers(self,corpus,start=3,end=11):
        dictionary, bow_corpus = self.BoW_vectorizing(corpus)

        X = [i for i in range(start,end)]
        Y_wob = []
        Y_tfidf = []

        # Calculate coherence scores
        for i in range(start,end):
            lda_bow = self.LDA_by_BoW(i,corpus)[0]
            lda_tfidf = self.LDA_by_Tfidf(i,corpus)[0]
            coherence_bow = CoherenceModel(model=lda_bow, texts=corpus, dictionary=dictionary, coherence='c_v')
            coherence_tfidf = CoherenceModel(model=lda_tfidf, texts=corpus, dictionary=dictionary, coherence='c_v')
            Y_wob.append(coherence_bow.get_coherence())
            Y_tfidf.append(coherence_tfidf.get_coherence())

        #draw figure
        plt.plot(X,Y_wob)
        plt.plot(X,Y_tfidf)
        plt.title('Coherence Values of LDA Model with Different numbers of topics',fontsize=20)
        plt.xlabel('The Number of Topic')
        plt.ylabel('Coherence Value')
        plt.legend(['Bag-of-words','Tf-Idf'])
        plt.show()        
        return

