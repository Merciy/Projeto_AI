import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords



 # CONFIG #
ARQUIVO = 'HateBRXplain.csv' 
NUM_COMENTARIOS = 10
LIMIAR_SIMILARIDADE = 0.3 # quanto tem que ser parecido as palavras para serem ====


df = pd.read_csv(ARQUIVO)
comentarios = df['comment'].head(NUM_COMENTARIOS)


stop_words = set(stopwords.words('portuguese'))
comentarios_processados = comentarios.apply(lambda x: ' '.join([word for word in x.lower().split() if word not in stop_words])) ## TIRA O A DE ESSAS COISAS DE LIGACAO OU DE COMECO
# Vetorizacao dos comentarios 
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(comentarios_processados)

similarity_matrix = cosine_simigilarity(tfidf_matrix)
# criacao dos grafos 
G = nx.Graph()
for i in range(len(comentarios)):
    for j in range(i + 1, len(comentarios)):
        if similarity_matrix[i, j] > LIMIAR_SIMILARIDADE:
            G.add_edge(i, j, weight=similarity_matrix[i, j])
# Desenho dos grafoss
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42, k=0.6)
nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_color='black', edge_color='gray', width=0.5)
plt.title('Grafo de Similaridade dos Comentários')
plt.show()
# mostrar commentarios smililares
for i in range(len(comentarios)):
    similar_indices = [j for j in range(len(comentarios)) if similarity_matrix[i, j] > LIMIAR_SIMILARIDADE and i != j]
    if similar_indices:
        print(f"Comentário {i + 1}: {comentarios.iloc[i]}")
        print("Comentários similares:")
        for j in similar_indices:
            print(f"  - Comentário {j + 1}: {comentarios.iloc[j]} (similaridade: {similarity_matrix[i, j]:.2f})")
        print("\n")
# Exibição do grafo de similaridade    