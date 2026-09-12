import os
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import torch.nn.functional as F

def build_concept_hierarchy(dict_dir, n_clusters=8):
    matrix_path = os.path.join(dict_dir, 'biomedclip_phrase_level_concept_matrix.pt')
    metadata_path = os.path.join(dict_dir, 'biomedclip_phrase_level_metadata.csv')
    
    T_phrases = torch.load(matrix_path, map_location='cpu')
    phrase_metadata = pd.read_csv(metadata_path)
    
   
    parent_concepts = phrase_metadata['parent_concept'].unique()
    parent_embeddings = []
    parent_mapping = []
    
    for concept in parent_concepts:
        idxs = phrase_metadata[phrase_metadata['parent_concept'] == concept]['phrase_index'].values
        concept_emb = T_phrases[idxs].mean(dim=0) 
        concept_emb = F.normalize(concept_emb, p=2, dim=0) 
        parent_embeddings.append(concept_emb)
        parent_mapping.append(concept)
        
    parent_embeddings = torch.stack(parent_embeddings).numpy()
    
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    cluster_labels = clustering.fit_predict(parent_embeddings)
    
    
    hierarchy_df = pd.DataFrame({
        'parent_concept': parent_mapping,
        'macro_cluster': cluster_labels
    })
    
    print(f"Created {n_clusters} clinical macro-cluster for the {len(parent_concepts)} concepts.")
    return hierarchy_df, parent_embeddings