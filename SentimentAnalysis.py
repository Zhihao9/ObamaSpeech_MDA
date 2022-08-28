import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd

nltk.download(['punkt','vader_lexicon'])

# All Operations related to sentiment analysis
class SentimentAnalysis(object):
    def __init__(self,speech):
        self.speech = speech

    def preprocess(self):
        split_speeches = []
        try:
            i = 0
            for s in self.speech:
                split_speeches.append(sent_tokenize(s))
                i = i+1
        except:
            print('errors at:',i)
            print(s)

        return split_speeches
    
    # To get the sentiment socores of sentences in each speech
    def sentiment_score(self,speeches):
        result = []
        sia = SentimentIntensityAnalyzer()
        for speech in speeches:
            scores = [] 
            for sentence in speech:
                tmp = sia.polarity_scores(sentence)
                scores.append(tmp['compound'])
            result.append(scores)
        return result

    # To calculate the proportions of different sentiment classes
    def senti_proportion(self,scores):
        res = []
        for score in scores:
            prop = [0 for i in range(5)]
            for s in score:
                if s<-0.7: # Very Negative
                    prop[0] = prop[0] + 1
                elif s<-0.2:  # Negative              
                    prop[1] = prop[1] + 1
                elif s < 0.2: # Neutral
                    prop[2] = prop[2] + 1
                elif s < 0.7:  # Positive
                    prop[3] = prop[3] + 1
                else:         # Very Positive
                    prop[4] = prop[4] + 1

            prop = [prop[i]/len(score) for i in range(5)]
            res.append(prop)

        return res

    # To show the Sentiment Distribution or proportions of each speech
    def draw_proportion(self,prop):
        very_neg = []
        neg = []
        neutral = []
        pos = []
        very_pos = []
        for p in prop:
            very_neg.append(p[0])
            neg.append(p[1])
            neutral.append(p[2])
            pos.append(p[3])
            very_pos.append(p[4])

        Y = [very_neg,neg,neutral,pos,very_pos]
        X = [i for i in range(len(prop))]

        plt.stackplot(X,Y,labels=['Very Negative','Negative','Neutral','Positive','Very Positive'])
        plt.ylim([0,1])
        plt.xlabel('Speeches')
        plt.ylabel('Proportion')
        plt.title('The Sentiment Proportion of Each Speech',fontsize=20)
        plt.legend()
        plt.show()
