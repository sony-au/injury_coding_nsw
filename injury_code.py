# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:48:40 2024

@author: sjufri
"""

import pandas as pd
import numpy as np
import torch
#import h5py

from sentence_embedding import calculate_cosine_matrix

def nature_injury_code(scenario_embedding,
                       n):
    '''
    This is function to obtain the TOOCS nature of injury code for an accident/injury
    scenario_input: the sentence embedding for the accident scenario and the resulting injury
    n: the top n coding you wish to display
    '''
    # load nature of injury data
    nature_df = pd.read_parquet('./dataset/nature_embeddings.parquet')
    nature_df['Embedding'] = nature_df['Embedding'].apply(lambda x: torch.from_numpy(x))
    
    # Convert embeddings from DataFrame to numpy array
    embedding_matrix = np.vstack(nature_df['Embedding'].values)
    
    # Calculate cosine similarity for all embeddings
    cosine_sims = calculate_cosine_matrix(embedding_matrix, scenario_embedding[0])
    
    nature_df['CosineSimilarity']=cosine_sims
    sorted_nature_df=nature_df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_nature_df.loc[:,['Code','BodilyLocation','Description']]

def body_injury_code(scenario_embedding,
                     n):
    '''
    This is function to obtain the TOOCS body location of injury code for an accident/injury
    scenario_embedding: the sentence embedding for the accident scenario and the resulting injury
    n: the top n coding you wish to display
    '''
    # load body of injury data
    body_df = pd.read_parquet('./dataset/body_embeddings.parquet')
    body_df['Embedding'] = body_df['Embedding'].apply(lambda x: torch.from_numpy(x))
    
    # Convert embeddings from DataFrame to numpy array
    embedding_matrix = np.vstack(body_df['Embedding'].values)
    
    # Calculate cosine similarity for all embeddings
    cosine_sims = calculate_cosine_matrix(embedding_matrix, scenario_embedding[0])
    
    body_df['CosineSimilarity']=cosine_sims
    sorted_body_df=body_df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_body_df.loc[:,['Code','BodyPart','Description']]

def mechanism_injury_code(scenario_embedding,
                          n):
    '''
    This is function to obtain the TOOCS mechanism of injury code for an accident/injury
    scenario_embedding: the sentence embedding for the accident scenario and the resulting injury
    n: the top n coding you wish to display
    '''
    # load mechanism of injury data
    mech_df = pd.read_parquet('./dataset/mech_embeddings.parquet')
    mech_df['Embedding'] = mech_df['Embedding'].apply(lambda x: torch.from_numpy(x))
    
    # Convert embeddings from DataFrame to numpy array
    embedding_matrix = np.vstack(mech_df['Embedding'].values)
    
    # Calculate cosine similarity for all embeddings
    cosine_sims = calculate_cosine_matrix(embedding_matrix, scenario_embedding[0])
    
    mech_df['CosineSimilarity']=cosine_sims
    sorted_mech_df=mech_df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_mech_df.loc[:,['Code','Mechanism','Description']]

def agency_injury_code(agency_embedding,
                       n):
    '''
    This is function to obtain the TOOCS agency of injury code for an accident/injury
    agency_embedding: the sentence embedding for the agency of the injury
    n: the top n coding you wish to display
    '''
    # load agency of injury data
    agency_df = pd.read_parquet('./dataset/agency_embeddings.parquet')
    agency_df['Embedding'] = agency_df['Embedding'].apply(lambda x: torch.from_numpy(x))
    
    # Convert embeddings from DataFrame to numpy array
    embedding_matrix = np.vstack(agency_df['Embedding'].values)
    
    # Calculate cosine similarity for all embeddings
    cosine_sims = calculate_cosine_matrix(embedding_matrix, agency_embedding[0])
    
    agency_df['CosineSimilarity']=cosine_sims
    sorted_agency_df=agency_df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_agency_df.loc[:,['Code','Agency','Description']]

def icd_code(scenario_embedding,
             n, 
             df):
    '''
    This is function to obtain the ICD injury code for an accident/injury
    scenario_input: the sentence embedding for the accident scenario and the resulting injury
    n: the top n coding you wish to display
    df: the icd dataframe
    '''
    # Convert embeddings from DataFrame to numpy array
    embedding_matrix = np.vstack(df['Embedding'].values)
    
    # Calculate cosine similarity for all embeddings
    cosine_sims = calculate_cosine_matrix(embedding_matrix, scenario_embedding[0])
    
    df['CosineSimilarity']=cosine_sims
    sorted_df=df.sort_values(by=['CosineSimilarity'], ascending=False).head(n)
    
    return sorted_df.loc[:,['Code', 'LongDescription']]