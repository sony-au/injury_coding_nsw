# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:42:47 2024

@author: 231829
"""

import pandas as pd
import numpy as np
import torch

from sentence_embedding import create_embedding, calculate_cosine_matrix

def find_industry_code(user_input,
                       n,
                       tokenizer, 
                       model):
    '''
    This is function to obtain the ANZSIC Industry Code for an occupation
    user_input: the industry you are searching for
    n: the top n coding you wish to display
    tokenizer: the tokenizer of the LLM model
    model: the LLM model used
    '''
    # load industry data
    industry_df = pd.read_parquet('./dataset/anzsic_2006_embeddings.parquet')
    industry_df['Embedding'] = industry_df['Embedding'].apply(lambda x: torch.from_numpy(x))
    
    # create embedding and calculate cosine similarity
    scenario_embedding=create_embedding(user_input, tokenizer, model)
    
    # Convert embeddings from DataFrame to numpy array
    embedding_matrix = np.vstack(industry_df['Embedding'].values)
    
    # Calculate cosine similarity for all embeddings
    cosine_sims = calculate_cosine_matrix(embedding_matrix, scenario_embedding[0])
    
    industry_df['CosineSimilarity']=cosine_sims
    sorted_industry_df=industry_df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_industry_df.loc[:,['Code','Division', 'SubDivision', 'Class', 'Description',]]

def find_wic_code(user_input,
                  n,
                  tokenizer, 
                  model):
    '''
    This is function to obtain the WIC Industry Code for an employer
    user_input: the industry you are searching for
    n: the top n coding you wish to display
    tokenizer: the tokenizer of the LLM model
    model: the LLM model used
    '''
    # load wic data
    wic_df = pd.read_parquet('./dataset/wic_embeddings.parquet')
    wic_df['Embedding'] = wic_df['Embedding'].apply(lambda x: torch.from_numpy(x))
    
    # create embedding and calculate cosine similarity
    scenario_embedding=create_embedding(user_input, tokenizer, model)
    
    # Convert embeddings from DataFrame to numpy array
    embedding_matrix = np.vstack(wic_df['Embedding'].values)
    
    # Calculate cosine similarity for all embeddings
    cosine_sims = calculate_cosine_matrix(embedding_matrix, scenario_embedding[0])
    
    wic_df['CosineSimilarity']=cosine_sims
    sorted_wic_df=wic_df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_wic_df.loc[:,['Code', 'WIC Description', 'WIC Rate','Dust Diseases Contribution (incl. GST)']]

