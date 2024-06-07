import io
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


def read_local_file(file_path):
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])
    elif file_path.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])
    else:
        with open(file_path, 'rb') as f:
            file_stream = io.BytesIO(f.read())
            try:
                return file_stream.read().decode('utf-8')
            except UnicodeDecodeError:
                try:
                    return file_stream.read().decode('latin-1')
                except UnicodeDecodeError:
                    return file_stream.read().decode('iso-8859-1')


def resume_to_vector(index, file_path):
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
    content = read_local_file(file_path)
    resume_id = os.path.basename(file_path)
    if content:
        vectorstore.add_texts(texts=[content], metadatas=[{"id": resume_id}])
        print("Files uploaded to Pinecone.")
    else:
        print("No content in this file.")
    return content


def find_similar_resumes(index, content, top_k=5):
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
    results = vectorstore.similarity_search(content, top_k)

    return results


def chat_similar_resumes(index, content):
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
    llm = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    query = f"Find similar resumes and highlight the similarities and reasons for the similarities: {content}"
    results = qa.run(query)

    return results
