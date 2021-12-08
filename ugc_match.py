#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:21:24 2021

@author:    Md Rayhanul Masud
            University of California Riverside
"""

from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
from os.path import exists
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import warnings
warnings.filterwarnings('ignore')

SUB_REDDIT_MAPPING = {
    'datascience': 1,
    'learnprogramming': 2,
    'learnpython': 3,
    'programmerhumor': 4    
}

DATASET_FILE_NAME='dataset.csv'


def read_data(filename, cols):
        
    print()
    df = pd.read_csv(filename, names=cols)
    
    df['pd_timeStamp'] = pd.to_datetime(df['timeStamp'], 
                                        format='%Y/%m/%d-%H:%M:%S')
    df['day_period'] = (df['pd_timeStamp'].dt.hour % 24 + 4) // 4
    
    
    df['day_period'] = df['day_period'].astype('int')
    print(df[ [ 'day_period', 'timeStamp'] ].head())
    
    # removing timestamp
    df.drop(['pd_timeStamp', 'timeStamp'], axis=1, inplace=True)
    
    
    return df



def get_url_count(text):
    
    url_list_http=re.findall(r'(http[|s]?://\S+)', text)
    # print(url_list_http)
    
    url_list_www=re.findall(r'(www.\S+)', text)
    # print(url_list_www)
    
    return len(url_list_http) + len(url_list_www)


def textSimilarity(text1, text2): # get the similarity score.
    """Compare two texts using BERT model. 
    
    Returns the Cosine Similarity of the embeddings obtained by BERT.
    """
    
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    embeddings1 = model.encode([text1], convert_to_tensor=True)
    embeddings2 = model.encode([text2], convert_to_tensor=True)

    #Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1,embeddings2)
    # cosine_scores
    return cosine_scores[0].item()
    

def get_clean_text(text):
    
    text = re.sub('[()!?]', ' ', text)
    text = re.sub('\[.*?\]',' ', text)
    text = re.sub("[^a-z0-9]"," ", text) # non alpha-numerics
                  
    return text



class DataPuller:
    
    def __init__(self):
        
        self._threads=None
        self._comments=None
        self._profile_matching_matrix=[]
        self._author_profiles_map={}
        self._author_author_profile_map={}
        
        
        self.thread_filename='threads.csv'
        self.comment_filename='comments.csv'
        
        self.THREAD_COLS=['index', 'subreddit', 'id', 'author_name', 'title',
             'is_self',	'selftext',	'url', 'commsNum', 'timeStamp']
        self.COMMENT_COLS=['index', 'id',	'comment_id','author_name',	
                           'comment','timeStamp','subreddit']
        self.MINIMUM_COMMENTS=5
        self.DELETED_AUTHOR_NAME='[deleted]'
        self._pair_data_list=[]
        self._total_matching_pair=0
        self._users_found_matching=set()
        
    
    def _read_threads_comments(self):
        
        self._threads=read_data(self.thread_filename, self.THREAD_COLS)
        print('Number of threads', self._threads.shape)
        print('Number of unique authors', len(self._threads['author_name'].unique()))
    
        self._comments=read_data(self.comment_filename, self.COMMENT_COLS)
        print('Number of posts/comments', self._comments.shape)
        
        self.show_stats_of_data()
        
    def show_stats_of_data(self):
        
        print("From Threads")
        for subreddit in SUB_REDDIT_MAPPING:
            
            author_count=len(self._threads[self._threads['subreddit']==subreddit]['author_name'].unique())
            print(subreddit, author_count)
        
        print()
        
        print("From Comments")
        for subreddit in SUB_REDDIT_MAPPING:
            
            author_count=len(self._comments[self._comments['subreddit']==subreddit]['author_name'].unique())
            print(subreddit, author_count)
            

    # Creating author profile for single subreddit
    def _create_profile(self, author_name, subreddit, threads, comments):
        
        raw_text=""
        clean_text=""
        url_count=0
        punctuation_count=0
        day_period=0
        
        
        if threads.shape[0] > 0:
            
            raw_text=raw_text + ' '.join(list(threads['title'])).lower()
            raw_text=raw_text + ' '.join(list(threads['selftext'].astype(str))).lower()
            
            # count url in thread
            url_count=url_count + threads[threads['is_self'] == False].shape[0]
            # posting time
            day_period=threads['day_period'].mean()
            
        if comments.shape[0] > 0:
    
            raw_text=raw_text + ' '.join(list(comments['comment'])).lower()
            
            # posting time
            day_period=int((day_period+comments['day_period'].mean())//2)
            
        url_count=url_count + get_url_count(raw_text)
        clean_text=get_clean_text(raw_text)
        punctuation_count=len(raw_text)-len(clean_text)
        
        profile={}
        profile['author_name']=author_name
        profile['subreddit']=subreddit
        profile['text']=clean_text
        profile['url_count']=url_count
        profile['punctuation_count']=punctuation_count
        profile['day_period']=day_period
        
        # print(profile)
        
        return profile


    # Creating author profiles for subreddits where he has contents
    def _get_profiles(self, author_name, threads, comments):
        
        profiles=[]
        if threads.shape[0] > 0:
            # print(author_name)
            subreddit_list=threads['subreddit'].unique()
            
            # print(subreddit_list)
            
            for subreddit in subreddit_list:
               
                profile=self._create_profile(author_name, subreddit, 
                               threads[(threads['subreddit']==subreddit)], 
                               comments[(comments['subreddit']==subreddit)])                                            # need to check
                
                profiles.append(profile)
                # break
        else:
            subreddit_list=comments['subreddit'].unique()
            
            # print(subreddit_list)
            
            for subreddit in subreddit_list:
                
                filtered_subreddit_specific=comments[(comments['subreddit']==subreddit)]
                if filtered_subreddit_specific.shape[0] > self.MINIMUM_COMMENTS:
                    profile=self._create_profile(author_name, subreddit, 
                                    threads, 
                                    filtered_subreddit_specific)
                    
                    profiles.append(profile)
                # break 
                    
        return profiles
    
        
    # Profile matching matrix
    def _init_profile_matching_matrix(self):
        
        unique_authors_threads=list(self._threads['author_name'].unique())
        # unique_authors_comments=list(self._comments['author_name'].unique())
        unique_authors_comments=[]
        agg=unique_authors_threads+unique_authors_comments
        unique_authors=dict.fromkeys(agg).keys()
        
        profile_index=-1
        
        for author_name in unique_authors:
            
            if author_name == self.DELETED_AUTHOR_NAME:
                continue
            
            profiles=self._get_profiles(author_name,            
                        self._threads[self._threads['author_name'] == author_name],
                        self._comments[self._comments['author_name'] == author_name])
            
            # if profile_index == 0:
            #     break
            
            for profile in profiles:
                profile_index=profile_index+1
                self._author_profiles_map[profile_index]=profile
                
                if author_name not in self._author_author_profile_map:
                    self._author_author_profile_map[author_name]=[profile_index]
                else:
                    self._author_author_profile_map[author_name].append(profile_index)
                    
        print(len(self._author_profiles_map.keys()), 'Profiles found')
        print(profile_index+1==len(self._author_profiles_map.keys()))
        
        
        total_profile_count=profile_index+1
        
        for profile_index, profile in self._author_profiles_map.items():
            
            profile_match_row=[False]*total_profile_count
            
            author_name=profile['author_name']
            profile_index_list=self._author_author_profile_map[author_name]
            for matched_index in profile_index_list:
                profile_match_row[matched_index]=True
                
            self._profile_matching_matrix.append(profile_match_row)
        

    # display matching matrix
    def _show_profile_matchings(self):
    
        
        for row_index, row  in  enumerate(self._profile_matching_matrix):
            
            for col_index, matching_label in enumerate(row):
                
                if matching_label == True and row_index < col_index:
                    self._total_matching_pair=self._total_matching_pair+1
                    
                    # print(row_index, col_index)
                    profile1=self._author_profiles_map[row_index]
                    # print("Subreddit 1", profile1['author_name'], profile1['subreddit'])
                    profile2=self._author_profiles_map[col_index]
                    # print("Subreddit 2", profile2['author_name'], profile2['subreddit'])
                    
                    self._users_found_matching.add(profile1['author_name'])
                    self._users_found_matching.add(profile2['author_name'])
                    
                    # with open('check.csv', 'a') as fp:
                    #     fp.write("Subreddit 1 " + profile1['author_name'] + " " + profile1['subreddit'] + '\n')
                    #     fp.write("Subreddit 2 " + profile2['author_name'] + " " + profile2['subreddit'] + '\n')
                    #     fp.write('\n')
                    # print()
        print("total_matching_pair found", self._total_matching_pair)
        print(list(self._users_found_matching))
        print(len(list(self._users_found_matching)))
        print()
    
    
    def _get_pair_data(self, author_1, author_2, label):
        
        
        pair_data={}
        pair_data['author_1_name']=author_1['author_name']
        pair_data['author_1_subreddit']=author_1['subreddit']
        
        pair_data['author_2_name']=author_2['author_name']
        pair_data['author_2_subreddit']=author_2['subreddit']
        
        pair_data['url_count']=abs(author_1['url_count']-author_2['url_count'])
        pair_data['punctuation_count']=abs(author_1['punctuation_count']-author_2['punctuation_count'])
        pair_data['day_period']=abs(author_1['day_period']-author_2['day_period'])
        pair_data['text_similarity']=textSimilarity(author_1['text'],author_2['text'])
        
        
        
        # pair_data['author_2_text_score']=author_2['text']
        
        
        pair_data['label'] = label # labels are 1 or 0
        
        # print(pair_data)
        
        return pair_data
        
    
    def generate_labeled_data(self, num_random_samples=6):
        
        self._read_threads_comments()
        self._init_profile_matching_matrix()
        
        self._show_profile_matchings()
        
        # match_limit=self._total_matching_pair // 2
        # total_profile_count=len(self._author_profiles_map.keys())
        unmatch_limit=self._total_matching_pair * 5 
        # match_count=0
        unmatch_count=0
        
        self._users_found_matching
        
        for row_index, row  in  enumerate(self._profile_matching_matrix):
            
            for col_index, matching_label in enumerate(row):
                
                # if match_count > match_limit and unmatch_count > unmatch_limit:
                #     break
                
                if row_index < col_index:
                    
                    if matching_label == True :
            
                            # if matching_label == True:
                            #     match_count=match_count+1
                        
                            # if matching_label == False:
                            #     unmatch_count=unmatch_count+1
                        
                            
                            self._pair_data_list.append(
                                self._get_pair_data(self._author_profiles_map[row_index],
                                                self._author_profiles_map[col_index], 
                                                matching_label) 
                                )
                            
                            print(row_index, col_index)
                            print("added")
                
                    else:
                        if unmatch_count < unmatch_limit:
                            unmatch_count=unmatch_count+1
                            
                            self._pair_data_list.append(
                                self._get_pair_data(self._author_profiles_map[row_index],
                                                self._author_profiles_map[col_index], 
                                                matching_label) 
                                )
                            
                            print(row_index, col_index)
                            print("added in unmatch", unmatch_count)
                
                
                
            # if match_count > match_limit and unmatch_count > unmatch_limit:
            #         break
        
    
    
    def get_pair_data_list(self):
        
        return self._pair_data_list
    
def create_train_test_dataset(df, featureNames, labelName):
    
      df_feature= df[featureNames]
      df_label = df[[labelName]]
      x_features = df_feature.values
      y_labels = df_label.values
      x_train, x_test, y_train, y_test = train_test_split(x_features, y_labels, test_size=0.33, random_state=7, stratify=y_labels)

      return x_train, x_test, y_train, y_test
  
    
def fitModel(model_name, model, X_train, y_train, X_test, y_test, verbose = True):
    
    model.fit(X_train, y_train)        
    
    print(f"{model_name}: Score (roc_auc) = {model.score(X_train, y_train)}, Best Parameters= {model.best_params_}")
    
    print(f"{model_name}: Test Score (roc_auc) = {model.score(X_test,y_test)} ")   
    print(classification_report(y_true=y_test,y_pred=model.predict(X_test))) 
    plot_confusion_matrix(model, X_test, y_test) 
    plt.show() 


def train_model(model_name, x_train, x_test, y_train, y_test, cross_fold):
    
     
    if model_name == 'RF':
            param_grid  = {'n_estimators': [2,4,8,16,32,64,128,256,400], 'criterion':["gini", "entropy"], 'random_state': [42]}
            grid_er1 = GridSearchCV(estimator=ensemble.RandomForestClassifier(),param_grid =param_grid ,cv=cross_fold, scoring='roc_auc', n_jobs= -1)
            fitModel(f"{model_name}", grid_er1, x_train, y_train, x_test, y_test)
    elif model_name == 'AB':
            param_grid  = {'n_estimators': [2,4,8,16,32,64,128,256,400], 'learning_rate':[0.001,0.01,0.1,0.2] ,  'random_state': [42]}
            grid_er1 = GridSearchCV(estimator=ensemble.AdaBoostClassifier(),param_grid =param_grid ,cv=cross_fold, scoring='roc_auc', n_jobs= -1)
            fitModel(f"{model_name}", grid_er1, x_train, y_train, x_test, y_test)
    elif model_name == 'BC':
            param_grid  = {'n_estimators': [2,4,8,16,32,64,128,256,400], 'random_state': [42]}
            grid_er1 = GridSearchCV(estimator=ensemble.BaggingClassifier(),param_grid =param_grid ,cv=cross_fold, scoring='roc_auc', n_jobs= -1)
            fitModel(f"{model_name}", grid_er1, x_train, y_train, x_test, y_test)
 
    
if __name__ == '__main__':
    
    
    file_exists = exists(DATASET_FILE_NAME)
    
    if file_exists:
        df = pd.read_csv(DATASET_FILE_NAME)
        print('Dataset read done')
    
    else:
        print("No Dataset found")
        data_puller=DataPuller()
        data_puller.generate_labeled_data()
        
        data=data_puller.get_pair_data_list()
        print('Dataset generation done')
        
        df = pd.DataFrame.from_records(data)
        df = shuffle(df)
        print(df.head())
        
        df.to_csv('dataset.csv')
        print('Dataset write donne')

    
    featureNames = ['text_similarity', 'url_count', 'punctuation_count']
    labelName = 'label'
    
    
    model_names=['RF', 'AB', 'BC']
    cross_fold=10
    
    for model_name in model_names:
        
        x_train, x_test, y_train, y_test = create_train_test_dataset(
            df, featureNames, labelName)
        
        model=train_model(model_name, x_train, x_test, y_train, y_test, cross_fold)
    
    # classifier = Classifier(df, featureNames, labelName)
    # model = classifier.makeModel()
    
    # classifier.printFeatureRankings()
    # prinst(model)
    
    
    # threads=read_data(THREADS_FILENAME, THREADS_COL)
    # print(threads.shape)
    # print(len(threads['author_name'].unique()))
    # # print(len(threads['author_name'].unique()))
    # print(threads.groupby(['author_name', 'subreddit'])['subreddit'].count())
    # threads.groupby(['author_name', 'subreddit'])['subreddit'].count().to_csv('see.csv')
    
    
    # threads['subreddit']=threads['subreddit'].replace(list(SUB_REDDIT_MAPPING.keys()),
    #                                                    list(SUB_REDDIT_MAPPING.values()),
    #                                                    inplace=True)
    # print(len(threads['author_name'].unique()))
    # print(len(threads.shape))
    
    # print(threads.describe())
    # print(threads['subreddit'].unique())
    
    # comments=read_data(COMMENTS_FILENAME, COMMENTS_COL)
    # print(comments.describe())
    # print(comments['subreddit'].unique())
    
    # print(len(comments['author_name'].unique()))
    
    
    
    
    
    