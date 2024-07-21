import streamlit as st
from llm import llm, embeddings
from graph import graph

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate

# Create the Neo4jVector
neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              
    graph=graph,                             
    index_name="vector",                 
    node_label="Chunk",                      
    text_node_property="text",               
    embedding_node_property="embedding", 
    retrieval_query="""
RETURN
    node.text AS text,
    score,
    {
        id: node.id,
        source: node.source,
        page: node.page
    } AS metadata
"""
)

# Create the retriever
retriever = neo4jvector.as_retriever()

# Create the prompt
instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", instructions),
    ("human", "{input}"),
])

# Create the chain 
question_answer_chain = create_stuff_documents_chain(llm, prompt)
text_retriever = create_retrieval_chain(
    retriever,
    question_answer_chain,
)

# Create a function to call the chain
def get_chunk_text(input):
    return text_retriever.invoke({"input": input})