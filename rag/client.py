
from typing import Dict, List
from pathlib import Path

from chromadb import PersistentClient
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from litellm import completion

from rag.util_classes import RagError, RankOrder, Result
    
class RagClient():
    WAIT = wait_exponential(multiplier=1, min=10, max=240)
    DB_NAME = str(Path(__file__).parent / "preprocessed_db")
    COLLECTION_NAME = "knowledge_base"
    EMBEDDING_MODEL = "text-embedding-3-large"
    RETRIEVAL_K = 20
    FINAL_K = 10

    def __init__(self, model):
        self.model = model
        self.chroma = PersistentClient(path=self.DB_NAME)
        self.collection = self.chroma.get_or_create_collection(self.COLLECTION_NAME)
        self.openai = OpenAI()
    
    @retry(wait=WAIT)
    def _rerank(self, question, chunks):
        system_prompt = """
        You are a document re-ranker.
        You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
        The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
        You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
        Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
        """
        user_prompt = f"""The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, 
        from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.
        
        Here are the chunks:\n\n"""
        
        for index, chunk in enumerate(chunks):
            user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
        user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = completion(model=self.model, messages=messages, response_format=RankOrder)
        reply = response.choices[0].message.content
        order = RankOrder.model_validate_json(reply).order
        return [chunks[i - 1] for i in order]

    @staticmethod
    def _merge_chunks(chunks: List, reranked: List) -> List:
        merged = chunks[:]
        existing = [chunk.page_content for chunk in chunks]
        for chunk in reranked:
            if chunk.page_content not in existing:
                merged.append(chunk)
        return merged
    
    @retry(wait=WAIT)
    def _rewrite_query(self, question, history=[]):
        """
        Rewrite the user's question to be a more specific question that is more likely to surface 
        relevant content in the Knowledge Base.
        """
        
        message = f"""
        You are in a conversation with a user, answering questions about classical music and the company Popsical.
        You are about to look up information in a Knowledge Base to answer the user's question.

        This is the history of your conversation so far with the user:
        {history}

        And this is the user's current question:
        {question}

        Respond only with a short, refined question that you will use to search the Knowledge Base.
        It should be a VERY short specific question most likely to surface content. Focus on the question details.
        IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
        """
        response = completion(model=self.model, messages=[{"role": "system", "content": message}])
        return response.choices[0].message.content
    
    def _fetch_context_unranked(self, question):
        try:
            query = self.openai.embeddings.create(
                model=self.EMBEDDING_MODEL, 
                input=[question]
            ).data[0].embedding
            results = self.collection.query(query_embeddings=[query], n_results=self.RETRIEVAL_K)
        except Exception as e:
            print(e)
            raise RagError(e)

        chunks = []
        for result in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append(Result(page_content=result[0], metadata=result[1]))
        return chunks
    
    def fetch_context(self, question: str, history: str) -> List[Dict]:
        rewritten_question = self._rewrite_query(question, history)
        chunks1 = self._fetch_context_unranked(question)
        chunks2 = self._fetch_context_unranked(rewritten_question)
        chunks = self._merge_chunks(chunks1, chunks2)
        reranked = self._rerank(question, chunks)
        return reranked[:self.FINAL_K]
