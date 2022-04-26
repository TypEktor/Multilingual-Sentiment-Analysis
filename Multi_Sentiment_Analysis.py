import os
import pandas as pd
import numpy as np
import re
# https://github.com/davidmogar/cucco
from cucco import Cucco
import FunctionsMLSA as fu
import nltk
import string
from nltk import word_tokenize
from snowballstemmer import TurkishStemmer
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
pd.set_option('mode.chained_assignment', None)
from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.figure(dpi=1200)
from IPython.display import display
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from collections import Counter


class MultiSA:

    def __init__(self, method, directory):
        self.method = method
        self.directory = directory
        
    def GetData(self):
        
        directory  = self.directory + '/Data/'
        # Read xlsx files and create the one dataframe to rule them all (concatenate)
        filepaths = [directory + f for f in os.listdir(directory) if f.endswith('.xlsx')]
        df = pd.concat(map(pd.read_excel, filepaths))
        df.to_csv(directory+'PreCleaned Merged Tweets.csv', index=False)
        print('GetData - Path of the saved File: ' + directory)
        return df
        
    def Cleaning(self, df):
        print("Cleaning - The given method is: " + self.method)
        
        # function for remove stopwords, punctuations, numbers
        def text_preprocess(text):
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = re.sub('[0-9]+', '', text)
            text = [word for word in text.split() if word.lower() not in stop_word_list]
            return " ".join(text)
        
        def stemming_tokenizer(text): 
            stemmer = TurkishStemmer()
            return [stemmer.stemWord(w) for w in word_tokenize(text)]
        
        df = df.drop_duplicates(subset=['Text'])
        df = df.reset_index(drop=True)
        

        # Removing RT
        remove_rt = lambda x: re.sub('RT @\w+ '," ",x)
        df["TextCleaned"] = df['Text'].map(remove_rt)
        # Removing &amp; which is equal to &
        df['TextCleaned'] = df['TextCleaned'].replace(r'&amp;', '', regex=True)
        # Removing sites
        df['TextCleaned'] = df['TextCleaned'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
        # Removing @...
        df['TextCleaned'] = df['TextCleaned'].replace(r'@\S+', '', regex=True)
        
        normEng = Cucco()
        norms = ['remove_stop_words', 'replace_punctuation', 'remove_extra_whitespaces']
        
        # For turkish
        # WPT = nltk.WordPunctTokenizer()
        stop_word_list = nltk.corpus.stopwords.words('turkish')
        
        english = df[df['Language'] == 'en']
        turkish = df[df['Language'] == 'tr']
        
        if self.method == 'Vader':
            # Expand contractions
            english["TextCleaned"] = english['TextCleaned'].map(lambda x: fu.expandContractions(x))
            # removing stop words / punctuations / extra whitespaces
            english["TextCleaned"] = english['TextCleaned'].map(lambda x: normEng.normalize(x, norms))
            
            turkish['TextCleaned'] = turkish['TextCleaned'].apply(lambda x: ' '.join(stemming_tokenizer(x)))
            turkish['TextCleaned'] = turkish['TextCleaned'].apply(text_preprocess)
            
        elif self.method == 'Text Full Cleaned':
            # Lowercase
            english["TextCleaned"] = english['TextCleaned'].str.lower()
            turkish["TextCleaned"] = turkish['TextCleaned'].str.lower()
            # Expand contractions
            english["TextCleaned"] = english['TextCleaned'].map(lambda x: fu.expandContractions(x))
            # Normalisation
            english["TextCleaned"] = english['TextCleaned'].map(lambda x: normEng.normalize(x, norms))
            # Remove emojis
            english['TextCleaned'] = english['TextCleaned'].apply(fu.deEmojify)
            turkish['TextCleaned'] = turkish['TextCleaned'].apply(fu.deEmojify)
            # Stemming
            # https://www.nltk.org/_modules/nltk/stem/snowball.html
            english['TextCleaned'] = english['TextCleaned'].apply(lambda x: ' '.join(fu.stemming(x)))
            turkish['TextCleaned'] = turkish['TextCleaned'].apply(lambda x: ' '.join(stemming_tokenizer(x)))
            
            turkish['TextCleaned'] = turkish['TextCleaned'].apply(text_preprocess)
            
        elif self.method == 'Vader Lemmatization':
            # Lemmatize cleaned text (stem words)
            english['TextCleaned'] = english['TextCleaned'].astype(str).apply(fu.lemmatize_text)
            english['TextCleaned'] = english['TextCleaned'].apply(fu.collapse_list_to_string)
            
            # Can't apply lemmatization for the Turkish language - Stemming is considered more suitable
            turkish['TextCleaned'] = turkish['TextCleaned'].apply(lambda x: ' '.join(stemming_tokenizer(x)))
            turkish['TextCleaned'] = turkish['TextCleaned'].apply(text_preprocess)
            
        english.to_csv(self.directory+'/Data/Tweets English Cleaned.csv', index=False)
        turkish.to_csv(self.directory+'/Data/Tweets Turkish Cleaned.csv', index=False)

        # merge english and turkish     
        df1 = pd.concat([english, turkish])
        df1 = df1.reset_index(drop=True)
        
        # Strip text from spaces
        df1['TextCleaned'] = df1['TextCleaned'].map(lambda x: x.strip())
        # Save file
        df1.to_csv(self.directory+'/Data/Tweets Full Cleaned.csv', index=False)
        print('Cleaning - Path of the saved Files: ' + self.directory + '/Data/')
        return df1
        
    
########################################################################
########################################################################
##                               VADER                                ##
########################################################################
########################################################################

#VADER ( Valence Aware Dictionary for Sentiment Reasoning) is a model used for
#text sentiment analysis that is sensitive to both polarity (positive/negative)
#and intensity (strength) of emotion. It is available in the NLTK package and
#can be applied directly to unlabeled text data.


#VADER sentimental analysis relies on a dictionary that maps lexical features
#to emotion intensities known as sentiment scores. The sentiment score of a text
# can be obtained by summing up the intensity of each word in the text.

# For example- Words like ‘love’, ‘enjoy’, ‘happy’, ‘like’ all convey a positive sentiment. 
# Also VADER is intelligent enough to understand the basic context of these words, such as 
# “did not love” as a negative statement. It also understands the emphasis of capitalization 
# and punctuation, such as “ENJOY”

    def VADER(self, df):
        df= df[df['TextCleaned'].notna()]
        for index, row in tqdm(df['TextCleaned'].iteritems(), total=len(df)):
            # print(index)
            try:
                score = SentimentIntensityAnalyzer().polarity_scores(row)
            except:
                print("Tweet number can't be analysed and will be dropped") # "+index+"
                pass
            neg = score['neg']
            neu = score['neu']
            pos = score['pos']
            comp = score['compound']
            if neg > pos:
                df.loc[index, 'sentiment'] = "negative"
            elif pos > neg:
                df.loc[index, 'sentiment'] = "positive"
            else:
                df.loc[index, 'sentiment'] = "neutral"
            df.loc[index, 'neg'] = neg
            df.loc[index, 'neu'] = neu
            df.loc[index, 'pos'] = pos
            df.loc[index, 'compound'] = comp    
            
            sleep(0.001)
            
        english = df[df['Language'] == 'en']
        turkish = df[df['Language'] == 'tr']
        # Save file
        english.to_csv(self.directory+'/Data/Tweets English Analysed.csv', index=False)
        turkish.to_csv(self.directory+'/Data/Tweets Turkish Analysed.csv', index=False)
        df.to_csv(self.directory+'/Data/Tweets Full Analysed.csv', index=False)
        print('VADER - Path of the saved Files: ' + self.directory + '/Data/')
        return df
        
        
    def PiePlots(self, df):
        #Count_values for sentiment
        fu.count_values_in_column(df,"sentiment")
        # The slices will be ordered and plotted counter-clockwise.
        pichart = fu.count_values_in_column(df,"sentiment")
        TextPos = "Positive ("+str(pichart['Percentage'].iloc[0])+"%)"
        TextNeg = "Negative ("+str(pichart['Percentage'].iloc[2])+"%)"
        TextNeu = "Neutral ("+str(pichart['Percentage'].iloc[1])+"%)"
        # r'Positive ( %)',pichart['Percentage'].iloc[0], r'Neutral ( %)', r'Negative ( %)'
        labels = [TextPos, TextNeu, TextNeg]
        size = pichart["Percentage"]
        colors = ['green','grey','red']
        patches, texts = plt.pie(size, colors=colors, startangle=90)
        plt.legend(patches, labels, loc="best")
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        plt.axis('equal')
        plt.tight_layout()

        plt.savefig(self.directory+'/Output/Pie Chart VADER Full (Not cleaned text).png')
        plt.show()
        
    def WordCloud(self, df, NewPath):
        mask = np.array(Image.open(self.directory+'/Output/'+"cloud.png"))
        stopwords = set(STOPWORDS)
        wc = WordCloud(background_color="white",
                  mask = mask,
                  max_words=3000,
                  stopwords=stopwords,
                  repeat=True)
        wc.generate(str(df))
        wc.to_file(NewPath)
        print("Word Cloud Saved Successfully")
        # path="wc.png"
        display(Image.open(NewPath))
        
    def Stats(self, df):
        #Calculating tweet's lenght and word count
        df['text_len'] = df['TextCleaned'].astype(str).apply(len)
        df['text_word_count'] = df['TextCleaned'].apply(lambda x: len(str(x).split()))
        text_len = round(pd.DataFrame(df.groupby("sentiment").text_len.mean()),2)
        text_len.to_csv(self.directory +'/Output/Tur text len.csv')  
        text_word_count = round(pd.DataFrame(df.groupby("sentiment").text_word_count.mean()),2)
        text_word_count.to_csv(self.directory +'/Output/Tur text word count.csv')  
        
    def emojiAnalysis(self, df):
        
        tweets_contents = ','.join(df['Text'])
        all_emojis = fu.extract_emojis(tweets_contents)
        len(Counter(all_emojis).keys())
        Counter(all_emojis)
    
        common_emojis = Counter(all_emojis).most_common(30)
    
    
        vals = [x[1] for x in common_emojis]
        legends = [x[0] for x in common_emojis]
        plt.figure(figsize=(14,4))
        plt.ylim(0, 1000)
        plt.title('Top 30 Emojis in over 31k Related Tweets')
        #get rid of xticks
        plt.tick_params(
            axis='x',          
            which='both',      
            bottom=False,      
            top=False,         
            labelbottom=False)
        p = plt.bar(np.arange(len(legends)), vals, color="pink")
        # Make labels
        for rect1, label in zip(p, legends):
            height = rect1.get_height()
            plt.annotate(
                label,
                (rect1.get_x() + rect1.get_width()/2, height+5),
                ha="center",
                va="bottom",
                fontname='Segoe UI Emoji',
                fontsize=15
            )   
        plt.savefig(self.directory +'/Output/words_tweets.png')
        plt.show()
        
        