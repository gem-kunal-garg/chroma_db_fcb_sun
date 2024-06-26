import streamlit as st
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os


import os 
from dotenv import load_dotenv

load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")
# HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")# "sentence-transformers/all-mpnet-base-v2")


############################################### Install if using openAI
# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

import langchain.globals as lc_globals

lc_globals.set_verbose(False)  # or True if you want verbose output

################################################ Install if using hf
# from huggingface_hub import InferenceClient
# HF_TOKEN = "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"
# client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token= HF_TOKEN)


#################################################If you want to create new embeddings
# persist_directory = "chroma_db"

# vectordb = Chroma.from_documents(
#     documents=docs, embedding=embeddings, persist_directory=persist_directory
# )
# vectordb.persist()

################################################### If you already have embeddings
vectordb = Chroma(persist_directory="./chroma_fcdb", embedding_function=embeddings)


# import os
############################################ OpenAI api is used to answer the query based on the final context provided

# OPENAI_API_KEY = "sk-proj-9aacoPiOuZDIKBHvUoVqT3BlbkFJNCQL0QV1CSIyEwfvn8n5"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)


# chain = load_qa_chain(llm, chain_type="stuff",verbose=False)

# def get_answer_openai(question):
#     query = f"Use the following pieces of context which are selected from the financial reports of companies (Meta, Apple, Amazon, Alphabet) to answer the user's question, User's question is:{question}."
#     matching_docs = vectordb.similarity_search(query,k=2)#similarity_search(query)
#     st.write("matching_docs is :" ,matching_docs)
#     answer =  chain.run(input_documents=matching_docs, question=query)
#     # answer = chain.run(question=query)
#     return answer

chain = load_qa_chain(llm, chain_type="stuff")

def get_answer_openai(question):
    query = f"Use the following pieces of context which are selected from the financial reports of companies (Meta, Apple, Amazon, Alphabet) to answer the user's question, User's question is:{question}."
    matching_docs = vectordb.similarity_search(query, k=2)
    st.write("matching_docs is:", matching_docs)
    input_data = {"question": query, "input_documents": matching_docs}
    answer = chain.invoke(input=input_data)
    
    return answer["output_text"]

############################################ Hugging face api calls are used to answer the query based on the final context provided

# def get_answer_hf(context, question):
    
#     # client = InferenceClient(model="meta-llama/Llama-2-7b-chat-hf", token=HF_TOKEN)
#     res = "Response is empty"
#     # try:
#     res = client.text_generation(f"Use the following pieces of context which are selected from the financial reports of companies (Meta, Apple, Amazon, Alphabet, Netflix) to answer the user's question, User's question is:{question} and Context is :{context}.", max_new_tokens=250)
#     st.write("context is :",context)
#     # st.write("Question is :", question)
#     # st.write("result is :", res)
#     # print("THis is Get Answer function")
#     # logging.info("THis is get answer function from logging")
    
#     # except:
#     #     st.error('This is beyond the capability of the model to answer this right now.', icon="🚨")
        
#     return res

def hf_llm_qa(query):
    # matching_docs = vectordb.similarity_search_with_score(query,k=2) #similarity_search(query)
    # st.write(matching_docs)
    # matching_docs
    answer = get_answer_openai(query)
    # st.write("answer is :", answer)
    return answer



#streamlit application
st.title("Finance ChatBot")
def process_query(query):
    # Example logic to process the query
    return hf_llm_qa(query.lower())

def main():

    # Accept user input
    query = st.text_input('Enter your query here :')

    if st.button('Submit'):
        # Process the query
        result = process_query(query)
        
        # Display the result
        st.write('Response:')

        st.write(result)

if __name__ == '__main__':
    main()

    
