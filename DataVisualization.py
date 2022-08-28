# Import Packages
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

class DataVisualization(object):
    def __init__(self,df,topic_words,num_topic,topic_names):
        self.df = df
        self.topic_words = topic_words  # top words of topics
        self.num_topic = num_topic # hyperparameter: numbers of topics 
        self.topic_names = topic_names # the summarised name of each topic

    #Wordcloud for the topics
    def wordcloud(self,lda_model):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(stopwords=STOPWORDS,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=30,
                  colormap='tab10',
                  #color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

        topics = lda_model.show_topics(formatted=False,num_words=30)

        fig, axes = plt.subplots(2, 3, figsize=(10,10), sharex=True, sharey=True)

        for i, ax in enumerate(axes.flatten()):
            fig.add_subplot(ax)
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=500)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic' + str(i+1)+': '+self.topic_names[i], fontdict=dict(size=16))
            plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.show()

    # Draw the Sentiment Proportion of each topic
    def draw_senti_proportion(self):
        Y_veryneg = []
        Y_neg = []
        Y_neutral = []
        Y_pos = []
        Y_verypos = []
        for i in range(1,self.num_topic+1):
            tmp_df = self.df.loc[self.df['labels'] == i]
            very_neg = 0
            neg = 0
            neutral = 0
            pos = 0
            very_pos = 0
            for i, row in tmp_df.iterrows():
                very_neg += row['senti_proportions'][0]
                neg += row['senti_proportions'][1]
                neutral += row['senti_proportions'][2]
                pos += row['senti_proportions'][3]
                very_pos += row['senti_proportions'][4]

            very_neg = very_neg / len(tmp_df)
            neg = neg / len(tmp_df)
            neutral = neutral / len(tmp_df)
            pos = pos / len(tmp_df)
            very_pos = very_pos / len(tmp_df)

            Y_veryneg.append(very_neg)
            Y_neg.append(neg)
            Y_neutral.append(neutral)
            Y_pos.append(pos)
            Y_verypos.append(very_pos)
            
        X = ['Topic '+str(i)+'\n'+self.topic_names[i-1] for i in range(1,self.num_topic+1)]

        plt.bar(X,Y_veryneg,label='Very Negative',color='darkblue')
        plt.bar(X,Y_neg,label='Negative', bottom=Y_veryneg,color='royalblue')
        plt.bar(X,Y_neutral,label='Neutral',bottom=np.array(Y_veryneg)+np.array(Y_neg),color='gray')
        plt.bar(X,Y_pos,label='Positive',bottom=np.array(Y_veryneg)+np.array(Y_neg)+np.array(Y_neutral),color='wheat')
        plt.bar(X,Y_verypos,label='Very Positive',bottom=np.array(Y_veryneg)+np.array(Y_neg)+np.array(Y_neutral)+np.array(Y_pos),color='darkorange')

        plt.ylabel('Proportion')
        plt.xlabel('Topics')
        plt.title('The Sentiment Proportion of different Topic Labels',fontsize=20)
        plt.legend(loc="lower left",bbox_to_anchor=(1.0,0.5))
        plt.show()

    #show the proportions of the topics
    def Topic_Pie(self):
        count_df = self.df['labels'].value_counts().sort_index()
        labels = ['Topic '+str(i)+'\n'+self.topic_names[i-1] for i in range(1,self.num_topic+1)]
        plt.pie(count_df,labels=labels,autopct='%.2f%%')
        plt.title('The Proportion of Topics')
        plt.show()

    # Draw the time series of the topic nums of each topic
    def topic_nums_by_year(self):
        # Processing Data for use
        year = pd.to_datetime(self.df['dates']).dt.year
        year_df = pd.DataFrame({'year':year,'labels':self.df['labels']})
        group = year_df.groupby(['labels','year']).size().unstack(fill_value=0).stack().to_frame().reset_index()
        group.columns = ['labels','year','count']
        counts = []
        for i in range(1,self.num_topic+1):
            tmp = list(group[group['labels']==i]['count'])
            tmp.insert(1,0)  #There is no speech in 2003, so we add the new element by hand
            counts.append(tmp)
        
        X = [i for i in range(2002,2018)]
        sum_speech = np.zeros(len(X))
        for i in range(self.num_topic):
            sum_speech += np.array(counts[i])
        

        #Plot the number of speeches by years
        plt.plot(X,sum_speech)
        plt.vlines(2008,-1,max(sum_speech+1),color='r',label='Presidential Election')
        plt.vlines(2012,-1,max(sum_speech+1),color='r')
        plt.xlabel('Year')
        plt.ylabel('The Number of Speeches')
        plt.title('The Number of Speeches by years')
        plt.legend()
        plt.show()

        # Plot the Time Series of each topic in one figure
        for i in range(self.num_topic):
            plt.plot(X,counts[i],label='Topic '+str(i+1)+':'+self.topic_names[i])

        plt.legend()
        plt.xlabel('year')
        plt.ylabel('The Number of Speeches')
        plt.title('Time Series of the Topic Counts')
        plt.show()


        #plot the time series of each topic in each subplot
        fig, axes = plt.subplots(2,3,figsize=(16,9))
        for i, ax in enumerate(axes.flatten()):
            fig.add_subplot(ax)
            for j in range(self.num_topic):
                if i == j:
                    plt.plot(X,counts[j],color='r',label='Topic '+str(i+1))
                else:
                    plt.plot(X,counts[j],color='lightgrey')
            plt.ylabel('The Number of Speeches')
            plt.xticks([i for i in range(2002,2018,5)])
            plt.xlabel('Year')
            plt.title("Topic"+str(i+1)+': '+self.topic_names[i])
            plt.ylim(0,25)
            plt.legend()
        plt.show()

        return counts

    # Draw the plot to see the trend of the Topic numbers and external variabls
    def topic_external(self,yearly_counts,topic_i,exdata,ex_ylabel,title):
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        X = [i for i in range(2002,2018)]
        ax1.plot(X,yearly_counts[topic_i-1],label='Topic'+str(topic_i)+':'+self.topic_names[topic_i-1],color='b')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('The Number of Speeches',color='b')

        ax2.plot(exdata.iloc[:,0],exdata.iloc[:,1],color='r')
        ax2.set_ylabel(ex_ylabel, color='r')

        ax1.legend(loc=1)
        plt.title(title)
        plt.show()

    #draw the topic amount trend and its sentiment trend 
    def topic_senti_external(self,yearly_counts,topic_i, event_year,event_label):
        X = [i for i in range(2002,2018)]
        plt.plot(X,yearly_counts[topic_i-1],label='Topic'+str(topic_i)+': '+self.topic_names[topic_i-1])
        plt.vlines(event_year,-1,max(yearly_counts[topic_i-1])+1, label=event_label,color='r')
        plt.xlabel('Year')
        plt.ylabel('The Number of the Topic')
        plt.legend()
        plt.title('The Trend of Topic'+str(topic_i)+': '+self.topic_names[topic_i-1])
        plt.show()

        #draw heat map 
        care_df = self.df[self.df['labels']==topic_i]
        year = pd.to_datetime(care_df['dates']).dt.year
        year_df = pd.DataFrame({'year':year,'scores':self.df['senti_scores']})
        pos = []
        neg = []
        X = [i for i in range(2002,2018)]
        for year in X:
            tmp = year_df[year_df['year']==year]
            if not tmp.empty:
                sum_pos = 0
                sum_neg = 0
                for index,row in tmp.iterrows():
                    tmp_pos = [i for i in row['scores'] if i > 0.2]
                    tmp_neg = [i for i in row['scores'] if i < -0.2]
                    sum_pos += sum(tmp_pos) / (len(tmp_pos) + len(tmp_neg))
                    sum_neg += sum(tmp_neg) / (len(tmp_pos) + len(tmp_neg))
                pos.append(sum_pos / len(tmp))
                neg.append(sum_neg / len(tmp))
            else:
                pos.append(0)
                neg.append(0)


        heatmap_df = pd.DataFrame([pos,neg], index=['Positive','Negative'],columns=X)

        sns.heatmap(heatmap_df,cmap="RdBu_r",vmax=0.5,vmin=-0.5)
        plt.title('The Sentiment Heatmap of Topic'+str(topic_i)+': '+self.topic_names[topic_i-1])
        plt.show()
