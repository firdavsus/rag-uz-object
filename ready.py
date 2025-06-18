import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
import os
from FlagEmbedding import FlagReranker
from transformers import pipeline
import torch
from huggingface_hub import login

class RAG:
    def __init__(self, path_to_dict, path_to_embedings):
        self.path_to_dict=path_to_dict
        self.files = sorted(os.listdir(path_to_dict)) 
        self.embeddings = np.load(path_to_embedings).astype("float32")
        self.k=5
        self.rk=3

        ##for model##
        self.instructions="you are a lawyer, that is given a several laws and only based on them you should answer, the data will be given after this paragraph and at the end the question to you, (answer in the language the question was given providin a reference to the laws provided as a literate lawyer.)"
    
    def FAISS(self, query_text):
        dimension = self.embeddings.shape[1]

        # === Нормализация (если используешь cosine similarity) ===
        faiss.normalize_L2(self.embeddings)

        # === Создание квантизатора и индекса ===
        nlist = 512
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

        # === Обучение индекса ===
        sample_size = min(10000, len(self.embeddings))
        index.train(self.embeddings[:sample_size])

        # === Добавление эмбеддингов ===
        index.add(self.embeddings)

        # === Настройка количества просматриваемых кластеров ===
        index.nprobe = 20 

        # === Модель для получения эмбеддинга запроса ===
        model = SentenceTransformer("intfloat/multilingual-e5-base")

        query_embedding = model.encode("query: " + query_text, normalize_embeddings=True)
        query_embedding = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query_embedding)

        # === Поиск по индексу ===
        distances, indices = index.search(query_embedding, self.k)

        # === Загрузка исходных файлов ===
        faiss_res = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
            faiss_res.append(idx)

        print("faiss - done!")
        return faiss_res
    
    def ReRanker(self, query_text):
        list_of_files = self.FAISS(query_text)
        pairs = []
        texts = []

        for i in list_of_files:
            file_path = os.path.join(self.path_to_dict, self.files[i])
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
                pairs.append([query_text, text])

        model = FlagReranker(
            'BAAI/bge-reranker-large',
            use_fp16=True,
            devices=["cuda:0"],
        )

        scores = model.compute_score(pairs)

        scored_results = sorted(zip(scores, texts), key=lambda x: x[0], reverse=True)
        reranked=[]

        for i, (score, text) in enumerate(scored_results[:self.rk], 1):
            reranked.append(text)

        print("reranker - done!")
        return reranked
    
    def Model(self, question):
        
        info=self.ReRanker(question)
        info_2=""
        for i in info:
            info_2+=i
        info=info_2
        prompt=self.instructions
        prompt+=info
        prompt+=question
        pipe = pipeline(
            task="text-generation",
            model="google/gemma-1.1-7b-it",
            device_map="auto",
            torch_dtype=torch.float16
        )

        output = pipe(prompt, max_new_tokens=256, temperature=0.5)[0]['generated_text']

        return output

if __name__ == "__main__":
    login()
    model = RAG("./CleanedText/uz-latin/", "uz_embeddings.npy")
    
    question=[]
    while question!="exit":
        question=input("Question: ")
        answer=model.Model(question)
        print(answer)