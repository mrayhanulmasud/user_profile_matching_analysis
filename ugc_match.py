#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:21:24 2021

@author: masud
"""

# tokenize texts to get meaningful words
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from datetime import datetime
import re

MINIMUM_COMMENTS=5
DELETED_AUTHOR_NAME='[deleted]'

SUB_REDDIT_MAPPING = {
    'datascience': 1,
    'learnprogramming': 2,
    'learnpython': 3,
    'programmerhumor': 4    
}

THREADS_FILENAME="threads.csv"
THREADS_COL=['index', 'subreddit', 'id', 'author_name', 'title',
             'is_self',	'selftext',	'url', 'commsNum', 'timeStamp']


COMMENTS_FILENAME="comments.csv"
COMMENTS_COL=['index', 'id',	'comment_id','author_name',	'comment',	
             'timeStamp','subreddit']

def read_data(filename, cols):
    
    print()
    df = pd.read_csv(filename, names=cols)
    
    df['pd_timeStamp'] = pd.to_datetime(df['timeStamp'], format='%Y/%m/%d-%H:%M:%S')
    df['day_period'] = (df['pd_timeStamp'].dt.hour % 24 + 4) // 4
    
    
    df['day_period'] = df['day_period'].astype('int')
    print(df[ [ 'day_period', 'timeStamp'] ].head())
    
    # removing timestamp
    df.drop(['pd_timeStamp', 'timeStamp'], axis=1, inplace=True)
    
    
    return df

def show_stats_of_data(threads, comments):
    
    print("From Threads")
    for subreddit in SUB_REDDIT_MAPPING:
        
        author_count=len(threads[threads['subreddit']==subreddit]['author_name'].unique())
        print(subreddit, author_count)
    
    print()
    print("From Comments")
    for subreddit in SUB_REDDIT_MAPPING:
        
        author_count=len(comments[comments['subreddit']==subreddit]['author_name'].unique())
        print(subreddit, author_count)
        
def get_url_count(text):
    
    url_list_http=re.findall(r'(http[|s]?://\S+)', text)
    # print(url_list_http)
    
    url_list_www=re.findall(r'(www.\S+)', text)
    # print(url_list_www)
    
    return len(url_list_http) + len(url_list_www)

def get_clean_text(text):
    
    text = re.sub('[()!?]', ' ', text)
    text = re.sub('\[.*?\]',' ', text)
    text = re.sub("[^a-z0-9]"," ", text) # non alpha-numerics
                  
    return text

def use_bag_of_vector(thread_list):
    vectorizer = CountVectorizer()
    vectorizer.fit(text)
    
    # encode document
    vector = vectorizer.transform(text)
    
    # summarize encoded vector
    print(vector.shape)
    
    # print(type(vector))
    
    # print(vector.toarray())
    
    BoW_array = vector.toarray()
    
    print(BoW_array)

# Creating author profile for single subreddit
def create_profile(author_name, subreddit, threads, comments):
    
    raw_text=""
    clean_text=""
    url_count=0
    punctuation_count=0
    day_period=0
    
    
    if threads.shape[0] > 0:
        
        raw_text=raw_text + ' '.join(list(threads['title'])).lower()
        raw_text=raw_text + ' '.join(list(threads['selftext'])).lower()
        
        # count url in thread
        url_count=url_count + threads[threads['is_self'] == False].shape[0]
        # posting time
        day_period=threads['day_period'].mean()
        
    if comments.shape[0] > 0:

        raw_text=raw_text + ' '.join(list(comments['comment'])).lower()
        
        # posting time
        day_period=(day_period+comments['day_period'].mean())//2
        
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
    
    print(profile)
    return profile

# Creating author profiles for subreddits where he has contents
def get_profiles(author_name, threads, comments):
    
    profiles=[]
    if threads.shape[0] > 0:
        # print(author_name)
        subreddit_list=threads['subreddit'].unique()
        
        # print(subreddit_list)
        
        for subreddit in subreddit_list:
           
            profile=create_profile(author_name, subreddit, 
                           threads[(threads['subreddit']==subreddit)], 
                           comments[(comments['subreddit']==subreddit)])                                            # need to check
            
            profiles.append(profile)
            break
    else:
        subreddit_list=comments['subreddit'].unique()
        
        # print(subreddit_list)
        
        for subreddit in subreddit_list:
            
            filtered_subreddit_specific=comments[(comments['subreddit']==subreddit)]
            if filtered_subreddit_specific.shape[0] > MINIMUM_COMMENTS:
                profile=create_profile(author_name, subreddit, 
                                threads, 
                                filtered_subreddit_specific)
                
                profiles.append(profile)
            break 
                
    return profiles
    
        
# Profile matching matrix
def get_profile_matching_matrix(threads, comments):
    
    author_profiles_map={}
    author_author_profile_map={}
    
    unique_authors_threads=list(threads['author_name'].unique())
    unique_authors_comments=list(comments['author_name'].unique())
    agg=unique_authors_threads+unique_authors_comments
    unique_authors=dict.fromkeys(agg).keys()
    
    profile_index=-1
    
    for author_name in unique_authors:
        
        if author_name == DELETED_AUTHOR_NAME:
            continue
        
        profiles=get_profiles(author_name,            
                                threads[threads['author_name'] == author_name],
                                comments[comments['author_name'] == author_name])
        
        if profile_index == 0:
            break
        
        for profile in profiles:
            profile_index=profile_index+1
            author_profiles_map[profile_index]=profile
            
            if author_name not in author_author_profile_map:
                author_author_profile_map[author_name]=[profile_index]
            else:
                author_author_profile_map[author_name].append(profile_index)
                
    print('Profiles found', len(author_profiles_map.keys()))
    print(profile_index+1==len(author_profiles_map.keys()))
    
    profile_matching_matrix=[]
    total_profile_count=profile_index+1
    
    for profile_index in author_profiles_map:
        
        profile_match_row=[False]*total_profile_count
        
        author_name=author_profiles_map[profile_index]['author_name']
        profile_index_list=author_author_profile_map[author_name]
        for matched_index in profile_index_list:
            profile_match_row[matched_index]=True
            
        profile_matching_matrix.append(profile_match_row)
    
    return author_profiles_map, profile_matching_matrix

def show_profile_matchings(profile_matching_matrix):
    
    total_matching_pair=0
    a=set()
    for row_index, row  in  enumerate(profile_matching_matrix):
        
        for col_index, matching_label in enumerate(row):
            
            if matching_label == True and row_index < col_index:
                total_matching_pair=total_matching_pair+1
                
                # print(row_index, col_index)
                profile1=author_profiles_map[row_index]
                # print("Subreddit 1", profile1['author_name'], profile1['subreddit'])
                profile2=author_profiles_map[col_index]
                # print("Subreddit 2", profile2['author_name'], profile2['subreddit'])
                
                a.add(profile1['author_name'])
                a.add(profile2['author_name'])
                
                # with open('check.csv', 'a') as fp:
                #     fp.write("Subreddit 1 " + profile1['author_name'] + " " + profile1['subreddit'] + '\n')
                #     fp.write("Subreddit 2 " + profile2['author_name'] + " " + profile2['subreddit'] + '\n')
                #     fp.write('\n')
                # print()
    print("total_matching_pair found", total_matching_pair)
    print(list(a))
    print(len(list(a)))
    print()
    
    
    

if __name__ == '__main__':
    
    
    threads=read_data(THREADS_FILENAME, THREADS_COL)
    print('Number of threads', threads.shape)
    print('Number of unique authors', len(threads['author_name'].unique()))
    
    
    comments=read_data(COMMENTS_FILENAME, COMMENTS_COL)
    print('Number of posts/comments', comments.shape)
    
    # show_stats_of_data(threads, comments)
    
    
    author_profiles_map, profile_matching_matrix=get_profile_matching_matrix(
            threads, comments )
    
    show_profile_matchings(profile_matching_matrix)
    
                
    
    
    
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
    
    
    
    
    
    