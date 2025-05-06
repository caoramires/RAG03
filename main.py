# Aplicação RAG LGPD, desenvolvida com o GEM
"""
Programa RAG utilizando o model="google/flan-t5-large" com resultado de resposta correta porem com erros de
ortografia.
"""


from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

def carregar_e_dividir_pdf(caminho_arquivo):
    text = ""
    with open(caminho_arquivo, 'rb') as arquivo_pdf:
        leitor_pdf = PdfReader(arquivo_pdf)
        for pagina in leitor_pdf.pages:
            text += pagina.extract_text()

    # Agora, vamos dividir o texto em chunks menores.
    # Podemos definir um tamanho máximo para cada chunk (e.g., 512 caracteres)
    tamanho_chunk = 512
    chunks = [text[i:i + tamanho_chunk] for i in range(0, len(text), tamanho_chunk)]

    return chunks

def buscar_chunks_relevantes(pergunta, modelo_embeddings, indice, top_k=3):
    # 1. Gerar o embedding da pergunta
    embedding_pergunta = modelo_embeddings.encode([pergunta]).astype('float32')

    # 2. Realizar a busca no índice FAISS
    distancias, indices = indice.search(embedding_pergunta, top_k)

    # 3. Recuperar os chunks de texto correspondentes aos índices encontrados
    chunks_relevantes = [chunks_de_texto[i] for i in indices[0]]

    return chunks_relevantes

def gerar_resposta_otimizado(pergunta, chunks_contextuais, modelo_gerador, max_chunk_length=300, max_length_resposta=800):
    # Selecionar os 2 primeiros chunks (os mais relevantes)
    chunks_selecionados = chunks_contextuais[:2]

    # Truncar cada chunk para o tamanho máximo especificado
    chunks_truncados = [chunk[:max_chunk_length] for chunk in chunks_selecionados]

    # Construir o prompt para o modelo de linguagem
    contexto = "\n".join(chunks_truncados)
    prompt = f"Responda à seguinte pergunta com base no contexto fornecido:\n\n{pergunta}\n\nContexto:\n{contexto}\n\nResposta:"

    # Gerar a resposta usando o modelo
    try:
        resposta = modelo_gerador(prompt, max_length=max_length_resposta, num_return_sequences=1)[0]['generated_text']
    except Exception as e:
        return f"Ocorreu um erro na geração da resposta: {e}"
    return resposta

# Especifique o caminho para o seu arquivo PDF da LGPD
caminho_do_pdf = './data/L13709compilado.pdf'
chunks_de_texto = carregar_e_dividir_pdf(caminho_do_pdf)

print(f"O PDF foi dividido em {len(chunks_de_texto)} chunks.")
# Você pode imprimir os primeiros chunks para verificar
# print(chunks_de_texto[:2])

#_____________________________________________________________________________
# Carregue um modelo pré-treinado para embeddings
modelo_embeddings = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Gere os embeddings para cada chunk de texto
embeddings_dos_chunks = modelo_embeddings.encode(chunks_de_texto)

print(f"Formato da matriz de embeddings: {embeddings_dos_chunks.shape}")
# Cada linha da matriz 'embeddings_dos_chunks' corresponde ao embedding de um chunk.

#_______________________________________________________________________________________
# Dimensão dos embeddings (para o modelo 'paraphrase-multilingual-mpnet-base-v2')
dimensao_embedding = 768
# Número de chunks
num_chunks = len(embeddings_dos_chunks)

# Converter a lista de embeddings para um array numpy de ponto flutuante 32
embeddings_np = np.array(embeddings_dos_chunks).astype('float32')

# Criar o índice FAISS
indice = faiss.IndexFlatL2(dimensao_embedding)

# Adicionar os embeddings ao índice
indice.add(embeddings_np)

print(f"O índice FAISS contém {indice.ntotal} embeddings.")

#______________________________________________________________________________
# # Exemplo de uma pergunta do usuário
# pergunta_usuario = "Quais são os direitos dos titulares de dados segundo a LGPD?"
#
# # Buscar os chunks relevantes para a pergunta
# chunks_recuperados = buscar_chunks_relevantes(pergunta_usuario, modelo_embeddings, indice)
#
# print(f"Chunks relevantes encontrados para a pergunta: '{pergunta_usuario}'\n")
# for i, chunk in enumerate(chunks_recuperados):
#     print(f"Chunk {i+1}: {chunk[:100]}...\n") # Imprime os primeiros 100 caracteres de cada chunk

#___________________________________________________________________________________

# Carregar o modelo e o tokenizer do Flan-T5 (um modelo menor para demonstração)

modelo_gerador = pipeline("text2text-generation", model="google/flan-t5-large")

# Exemplo de uso da função de geração de respostas
pergunta_usuario = "Quais são os direitos dos titulares de dados segundo a LGPD?"
chunks_recuperados = buscar_chunks_relevantes(pergunta_usuario, modelo_embeddings, indice)
resposta_gerada = gerar_resposta_otimizado(pergunta_usuario, chunks_recuperados, modelo_gerador)

print(f"Pergunta: {pergunta_usuario}\n")
print(f"Resposta: {resposta_gerada}")