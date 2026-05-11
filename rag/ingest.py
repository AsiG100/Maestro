import json

from pathlib import Path
from openai import OpenAI
from litellm import completion
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from multiprocessing import Pool
from tenacity import retry, wait_exponential
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import numpy as np


load_dotenv(override=True)

MODEL = "openai/gpt-4.1-nano"

DB_NAME = str(Path(__file__).parent / "preprocessed_db")
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-large"
KNOWLEDGE_BASE_PATH = Path(__file__).parent / "data"
AVERAGE_CHUNK_SIZE = 100
WAIT = wait_exponential(multiplier=1, min=10, max=240)
WORKERS = 3

openai = OpenAI()


class Result(BaseModel):
    content: str
    metadata: dict


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document):
        metadata = {"type": document["type"]}
        return Result(
            content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


def fetch_documents():
    """A homemade version of the LangChain DirectoryLoader"""

    documents = []

    for file in KNOWLEDGE_BASE_PATH.glob("*.jsonl"):
        with open(file, "r", encoding="utf-8") as f:
            documents.append({"type": file.stem, "source": file.as_posix(), "text": f.read()})

    print(f"Loaded {len(documents)} documents")
    return documents


def make_prompt(document):
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
    You take a document and you split the document into overlapping chunks for a KnowledgeBase.

    The document is of type: {document["type"]}

    A chatbot will use these chunks to answer questions about classical music and Popsical's operations and history, and to help users find the right playlist for them or their child.
    You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
    This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
    There should be overlap between the chunks as appropriate; typically about 25% overlap or about 4 rows of jsonl, so you have the same text in multiple chunks for best retrieval results.

    For each chunk, you should provide a headline, a summary, and the original jsonl chunk.
    Together your chunks should represent the entire document with overlap.

    Here is the jsonl document:

    {document["text"]}

    Respond with the chunks.
    """


def make_messages(document):
    return [
        {"role": "user", "content": make_prompt(document)},
    ]


@retry(wait=WAIT)
def process_document(document):
    messages = make_messages(document)
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]


def create_chunks(documents):
    """
    Create chunks using a number of workers in parallel.
    If you get a rate limit error, set the WORKERS to 1.
    """
    chunks = []
    with Pool(processes=1) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks


def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(COLLECTION_NAME)

    texts = [chunk.content for chunk in chunks]
    emb = openai.embeddings.create(model=EMBEDDING_MODEL, input=texts).data
    vectors = [e.embedding for e in emb]

    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")

def visualize_vectors():
    chroma = PersistentClient(path=DB_NAME)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])
    vectors = np.array(result['embeddings'])
    documents = result['documents']
    metadatas = result['metadatas']
    doc_types = [metadata['type'] for metadata in metadatas]
    colors = [[
        'blue', 'green', 'red', 'orange',
        'purple', 'yellow', 'brown'
        ][[
        'composers', 'experience', 'pieces', 'playlists',
        'genres', 'instruments', 'popsical'
        ].index(t)] for t in doc_types]

    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    # Create the 2D scatter plot
    fig = go.Figure(data=[go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
        hoverinfo='text'
    )])

    fig.update_layout(title='2D Chroma Vector Store Visualization',
        scene=dict(xaxis_title='x',yaxis_title='y'),
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    fig.show()

if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
    visualize_vectors()
