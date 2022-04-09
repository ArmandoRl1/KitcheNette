import pickle
import pandas as pd
from tasks.data_loader import IngredientDataset
from tasks.run import *
from tasks.model import Model

with open('data/kitchenette_dataset.pkl', 'rb') as pickle_file:
    df = pickle.load(pickle_file)

with open('data/kitchenette_embeddings.pkl', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
pd.DataFrame(df.ingredients).to_csv('dataset_embeddings.csv')

pd.DataFrame(content).to_csv('kitchenette_embedding.csv')

# Loading the training data
df=pd.read_csv('data/kitchenette_pairing_scores.csv')

# Create dict of ingredient counts
ingredient_counts = {ingredient: count for ingredient, count in zip(pd.concat([df.ingr1, df.ingr2]), pd.concat([df['ingr1-count'], df['ingr2-count']]))}

# Read the Excel table
recetas_dominicanas = pd.read_excel('kitchenette_embedding.xlsx',sheet_name='recetario', usecols=[0,1,2,3])

ingredientes_unicos = recetas_dominicanas.INGREDIENTE.unique()
pares_posibles = np.unique([(x,y) for x in ingredientes_unicos for y in ingredientes_unicos if x!=y], axis=0)

# Create csv for all pairings in the dataset
pd.DataFrame({
    'ingr1': pares_posibles[:,0],
    'ingr2': pares_posibles[:,1],
    'pmi':'',
    'npmi':'',
    'pmi2':'',
    'pmi3':'',
    'ppmi':'',
    'co-occurence':0,
    'ingr1-count': [ingredient_counts[x] for x in pares_posibles[:, 0]],
    'ingr2-count': [ingredient_counts[x] for x in pares_posibles[:, 1]],
    'label':'unknown'
}).to_csv('data/kitchenette_unknown_pairings.csv')

# Now run the model with these parameters
'''
python.exe main.py --save-prediction-unknowns True --model-name kitchenette_pretrained.mdl --unknown-path ./data/kitchenette_unknown_pairings.csv --embed-path ./data/kitchenette_embeddings.pkl --data-path ./data/kitchenette_dataset.pkl
'''
