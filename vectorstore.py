from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import get_embeddings

from typing import List

# csv에 id,type,title,content,author,created_at

def load_documents() -> List[Document]:
    import pandas as pd

    df = pd.read_csv('./dataset.csv')
    return [
        Document(
            page_content=f"제목: {row['title']}\n\n내용: {row['content']}",
            metadata={
                'id': row['id'],
                'type': row['type'],
                'title': row['title'],
                'author': row['author'],
                'created_at': row['created_at']
            }
        ) for _, row in df.iterrows()
    ]


def split_docs(docs: List[Document]) -> List[Document]:
    """
    만일 문서가 충분히 크다면 청킹을 위해서 사용할 수 있습니다.
    """
    CHUNK_SIZE = 500
    OVERLAP_SIZE = 50
    

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE
    )

    return splitter.split_documents(docs)


def embedding(docs: List[Document]):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )
    return vectorstore


def save_vector_to_local(vectorstore):
    path_str = './exp-faiss'    
    vectorstore.save_local(path_str)


def load_vector_from_local():
    path_str = './exp-faiss'
    return FAISS.load_local(
        path_str,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

def init_vectorstore():
    docs = load_documents()
    vectorstore = embedding(docs)
    save_vector_to_local(vectorstore)
    return vectorstore