# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:07:05 2024

@author: sjufri
"""

import pandas as pd
import numpy as np
import torch
#import pickle

from sentence_embedding import create_embedding, calculate_cosine_matrix

def find_occupation_code(user_input,
                         n, 
                         tokenizer, 
                         model):
    '''
    This is function to obtain the ANZSCO Occupation Code for an occupation
    user_input: the occupation you are searching for
    n: the top n coding you wish to display
    tokenizer: the tokenizer of the LLM model
    model: the LLM model used
    '''
    # load occupation data
    occ_df = pd.read_parquet('./dataset/occ_embeddings.parquet')
    occ_df['Embedding'] = occ_df['Embedding'].apply(lambda x: torch.from_numpy(x))

    # create embedding and calculate cosine similarity
    scenario_embedding=create_embedding(user_input, tokenizer, model)
    
    # Convert embeddings from DataFrame to numpy array
    embedding_matrix = np.vstack(occ_df['Embedding'].values)
    
    # Calculate cosine similarity for all embeddings
    cosine_sims = calculate_cosine_matrix(embedding_matrix, scenario_embedding[0])
    
    occ_df['CosineSimilarity']=cosine_sims
    sorted_occ_df=occ_df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_occ_df.loc[:,['Code','Occupation','UnitGroup','MinorGroup','SubMajorGroup','MajorGroup']]