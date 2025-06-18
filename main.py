import pandas as pd
import customtkinter as ct
from tkinter import filedialog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

# Downloads necessários
nltk.download('stopwords')

# Configurações iniciais
ct.set_appearance_mode("dark")
ct.set_default_color_theme("green")

# Variáveis globais
df = None
NUM_COMENTARIOS = 10
LIMIAR_SIMILARIDADE = 0.3
stop_words = set(stopwords.words('portuguese'))

# Função para carregar o arquivo CSV
def carregar_csv():
    global df
    arquivo = filedialog.askopenfilename(
        filetypes=[("Arquivos CSV", "*.csv")],
        title="Selecione um arquivo CSV"
    )
    if arquivo:
        df = pd.read_csv(arquivo)
        label_status.configure(text=f"Arquivo carregado: {arquivo.split('/')[-1]}")

# Função para processar os comentários
def processar_comentarios():
    global df, NUM_COMENTARIOS
    if df is None or 'comment' not in df.columns:
        label_status.configure(text="Erro: CSV não carregado ou coluna 'comment' não encontrada.")
        return None, None, None

    comentarios = df['comment'].dropna().head(NUM_COMENTARIOS)

    comentarios_processados = comentarios.apply(lambda x: ' '.join(
        [word for word in x.lower().split() if word not in stop_words]
    ))

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(comentarios_processados)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return comentarios, similarity_matrix, tfidf_matrix

# Função para exibir o grafo
def exibir_grafo():
    comentarios, similarity_matrix, _ = processar_comentarios()
    if comentarios is None:
        return

    G = nx.Graph()
    for i in range(len(comentarios)):
        for j in range(i + 1, len(comentarios)):
            if similarity_matrix[i, j] > LIMIAR_SIMILARIDADE:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=0.6)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='orange',
            font_size=10, font_color='black', edge_color='gray', width=1.0)
    plt.title(f"Grafo de Similaridade ({NUM_COMENTARIOS} comentários)")
    plt.show()

# Função para mostrar comentários similares
def mostrar_similares():
    comentarios, similarity_matrix, _ = processar_comentarios()
    if comentarios is None:
        return

    texto_saida.delete("0.0", "end")  # Limpa saída anterior
    for i in range(len(comentarios)):
        similar_indices = [j for j in range(len(comentarios))
                           if similarity_matrix[i, j] > LIMIAR_SIMILARIDADE and i != j]
        if similar_indices:
            texto_saida.insert("end", f"Comentário {i + 1}: {comentarios.iloc[i]}\n")
            texto_saida.insert("end", "Comentários similares:\n")
            for j in similar_indices:
                texto_saida.insert("end", f"  - Comentário {j + 1}: {comentarios.iloc[j]} (similaridade: {similarity_matrix[i, j]:.2f})\n")
            texto_saida.insert("end", "\n")

# Função para atualizar o número de comentários a serem analisados
def atualizar_num_comentarios():
    global NUM_COMENTARIOS
    try:
        novo_num = int(entry_num_comentarios.get())
        if novo_num <= 0:
            raise ValueError
        NUM_COMENTARIOS = novo_num
        label_num_atual.configure(text=f"Número atual de comentários: {NUM_COMENTARIOS}")
    except ValueError:
        label_status.configure(text="Digite um número inteiro válido (>0).")

# Interface com CustomTkinter
janela = ct.CTk()
janela.title("Analisador de Similaridade de Comentários")
janela.geometry("800x650")

frame = ct.CTkFrame(janela)
frame.pack(padx=20, pady=20, fill="both", expand=True)

botao_carregar = ct.CTkButton(frame, text="Carregar CSV", command=carregar_csv)
botao_carregar.pack(pady=10)

# Campo para número de comentários
entry_num_comentarios = ct.CTkEntry(frame, placeholder_text="Número de comentários (ex: 10)", width=200)
entry_num_comentarios.pack(pady=5)

botao_atualizar_num = ct.CTkButton(frame, text="Atualizar Número de Comentários", command=atualizar_num_comentarios)
botao_atualizar_num.pack(pady=5)

label_num_atual = ct.CTkLabel(frame, text=f"Número atual de comentários: {NUM_COMENTARIOS}")
label_num_atual.pack(pady=5)

botao_grafo = ct.CTkButton(frame, text="Mostrar Grafo de Similaridade", command=exibir_grafo)
botao_grafo.pack(pady=10)

botao_similares = ct.CTkButton(frame, text="Mostrar Comentários Similares", command=mostrar_similares)
botao_similares.pack(pady=10)

label_status = ct.CTkLabel(frame, text="Nenhum arquivo carregado.")
label_status.pack(pady=10)

texto_saida = ct.CTkTextbox(frame, width=600, height=300)
texto_saida.pack(pady=10)

janela.mainloop()
