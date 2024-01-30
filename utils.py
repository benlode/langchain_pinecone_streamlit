import os
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings import SentenceTransformerEmbeddings
import openai
import streamlit as st

# Setting OpenAI API key from streamlit secret
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Set Pinecone API key from streamlit secret
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

index_name = 'langchain'

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=1536, 
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

# Connect to the index
index = pc.Index(name=index_name)

# Initialize the embeddings model
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Define your functions
def find_match(input):
    # Generate embeddings for the input
    input_em = embeddings_model.embed_documents([input])[0]
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
