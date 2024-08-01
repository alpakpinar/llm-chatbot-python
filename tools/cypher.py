import streamlit as st
from llm import llm
from graph import graph

from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate


CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about recipes and their ingredients.
Convert the user's question based on the schema.
In the generated Cypher query, please do not attempt exact string matches, but rather
search for string similarity, for example, using CONTAINS clause. And while comparing strings,
please convert both to lowercase.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.

Example Cypher Statements:

1. To find the ingredients on a given recipe:
MATCH (r:Recipe)-[:HAS_INGREDIENT]->(i:Ingredient)
WHERE toLower(r.name) CONTAINS toLower("recipe name")
RETURN i.name, i.quantity

Schema:
{schema}

Question:
{question}

Cypher Query:
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

# Create the Cypher QA chain
cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt,
)