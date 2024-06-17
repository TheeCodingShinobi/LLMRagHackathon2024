import os, json
import requests
import re
from flask import Flask, request, jsonify, send_from_directory

from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ChatMessageHistory
from langchain import OpenAI, LLMChain
from langchain_google_community import GoogleSearchAPIWrapper
# from langchain.utilities import GoogleSearchAPIWrapper
import getpass

    
os.environ["GOOGLE_CSE_ID"] =  os.environ.get("GOOGLE_CSE_ID","")
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY","")
    
app = Flask(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY","sk-")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
nvapi_key = os.environ.get("NVIDIA_API_KEY","nvapi-") #getpass.getpass("Enter your NVIDIA API key: ")
os.environ["NVIDIA_API_KEY"] = nvapi_key


OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DeploymentIP = "127.0.0.1"
AUDIO_SERVER_URL = "http://"+DeploymentIP+":5002/api/tts"

from typing import List, Union

import requests
from bs4 import BeautifulSoup

def html_document_loader(url: Union[str, bytes]) -> str:
    """
    Loads the HTML content of a document from a given URL and return it's content.

    Args:
        url: The URL of the document.

    Returns:
        The content of the document.

    Raises:
        Exception: If there is an error while making the HTTP request.

    """
    try:
        response = requests.get(url)
        html_content = response.text
    except Exception as e:
        print(f"Failed to load {url} due to exception {e}")
        return ""

    try:
        # Create a Beautiful Soup object to parse html
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style tags
        for script in soup(["script", "style"]):
            script.extract()

        # Get the plain text from the HTML document
        text = soup.get_text()

        # Remove excess whitespace and newlines
        text = re.sub("\s+", " ", text).strip()

        return text
    except Exception as e:
        print(f"Exception {e} while loading document")
        return ""
        
def create_embeddings(embedding_path: str = "./embed", urls = []):

    embedding_path = "./embed"
    print(f"Storing embeddings to {embedding_path}")

    if len(urls) == 0:
        # List of web pages containing NVIDIA Triton technical documentation
        urls = [
             "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html",
             "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html",
             "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html",
             "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_analyzer.html",
             "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html",
        ]

    documents = []
    for url in urls:
        document = html_document_loader(url)
        documents.append(document)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
    )
    texts = text_splitter.create_documents(documents)
    index_docs(url, text_splitter, texts, embedding_path)
    print("Generated embedding successfully")

def index_docs(url: Union[str, bytes], splitter, documents: List[str], dest_embed_dir) -> None:
    """
    Split the document into chunks and create embeddings for the document

    Args:
        url: Source url for the document.
        splitter: Splitter used to split the document
        documents: list of documents whose embeddings needs to be created
        dest_embed_dir: destination directory for embeddings

    Returns:
        None
    """
    embeddings = NVIDIAEmbeddings(model="nvolveqa_40k")
    
    for document in documents:
        texts = splitter.split_text(document.page_content)

        # metadata to attach to document
        metadatas = [document.metadata]

        # create embeddings and add to vector store
        if os.path.exists(dest_embed_dir):
            update = FAISS.load_local(folder_path=dest_embed_dir, embeddings=embeddings,
                                allow_dangerous_deserialization=True)
            update.add_texts(texts, metadatas=metadatas)
            update.save_local(folder_path=dest_embed_dir)
        else:
            docsearch = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
            docsearch.save_local(folder_path=dest_embed_dir)

from langchain_community.chat_message_histories import SQLChatMessageHistory


def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///ChatHistoryMemory.db")


TypesOfChats = [
    "MainChat", #Main Chat is Mandatory
    "AnalyticalChat", 
    "BuyingRecommendations", 
    "ForecastingChat"
]


ChatHistories = {}

for chat in TypesOfChats:
        ChatHistories[chat] = get_session_history(chat)
        
ContextPrompts = {}
#Data port of old Permaculture Agriculture application
ContextPrompts["MainChatContext"] = "You are a helpful and knowledgable agricultural expert. " + \
                                    "If the user asks for a topic unrelated to agriculture and/or climate change, kindly steer them towards the topic " + \
                                    "If the user asks for a topic unrelated to agriculture and/or climate change and cannot be sigued into the topic, politely decline their request " + \
                                    "The user may ask specific questions about permaculture. There is extra information in the context provided " + \
                                    "The context from the assistant may provide the current location of the user (through CurrentLocation), the beds of crops that the user is planning or tending to (through GeoJSONPins), and various plants that the user has scouted in the environment (through Biodiversity)  " + \
                                    "If the user asks about current actions on the farm, you can refer to the GeoJSONPins PlotPlans and BedNumber. The Vegetable will be listed, along with Start and endDates for each time " + \
                                    "Please be specific whenever possible, naming the state or region and plants that are native to it can help as examples. " + \
                                    "The day number will be 0-365 (e.g. April 1 ist 91), the app does not calculate Leap Years, but you can  " + \
                                    "Actions taken on a farm may include planting (where the current day the one of first 3 days in the Plot Plans), harvesting if it's one of the last 5 days in the Plot Plans, tending to plants/helpful growing advice if the plant is growing, or bed maintence if nothing is growing in the bed. " + \
                                    "A user asking for permaculture advice may not have room for a lot of actions, so around 3 tips or action points/beds will be suitable " + \
                                    "When answering questions related to biodiversity, focus on the importance of preserving and promoting the variety of species, habitats, and ecosystems in the environment. Provide suggestions on how the user can enhance biodiversity on their farm or in their local area, such as planting native species, creating habitats for wildlife, and implementing sustainable agricultural practices. "
                
search = GoogleSearchAPIWrapper()
zshot_tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]



##################################
### FLASK ENDPOINTS START HERE ###
##################################
@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    context_data = request.form.get("context_data")
    #print(context_data)
    chatgpt_response = send_question_to_chatgpt(question,context_data)
    chatgpt_response = chatgpt_response if chatgpt_response else "Sorry, I must have misunderstood. Could you repeat or rephrase the question?"
    audio_text = get_audio_text(chatgpt_response.replace('\n',''))

    response_data = {
        "chatgpt_response": chatgpt_response,
        "audio_text": audio_text
    }

    return jsonify(response_data)


@app.route("/askNVIDIA", methods=["POST"])
def askNVIDIA():
    question = request.form.get("question")
    chatType = "MainChat" #request.form.get("chat")
    if chatType in ChatHistories:
        ChatHistories[chatType].add_user_message(question);
    else:
        print(chatType + " not in Histories")
    
    prefix =  ContextPrompts["MainChatContext"] + """ 
You have access to the following tools:"""
    suffix = """Begin!"

    {chat_history}
    Question: """+question+"""
    """

    prompt = ZeroShotAgent.create_prompt(
        zshot_tools, 
        prefix=prefix, 
        suffix=suffix, 
        input_variables=["input", "chat_history"]
    )
    
    
    zshot_llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = zshot_llm_chain(prompt)
    
    # zshot_agent = ZeroShotAgent(llm_chain=zshot_llm_chain, tools=zshot_tools, verbose=True)
    #zshot_agent_chain = AgentExecutor.from_agent_and_tools(agent=zshot_agent, tools=zshot_tools, verbose=True, memory=memory)

    #agent_response = zshot_agent_chain.run(input=question, verbose = True);
    
    # chatgpt_response = send_question_to_chatgpt(question,context_data)
    # chatgpt_response = chatgpt_response if chatgpt_response else "Sorry, I must have misunderstood. Could you repeat or rephrase the question?"
    # audio_text = get_audio_text(chatgpt_response.replace('\n',''))
    #print(agent_response)
    
    if chatType in ChatHistories:
        ChatHistories[chatType].add_ai_message(response);
    else:
        print(chatType + " not in Histories")
    response_data = {
        "agent_response": response.content,
        # "audio_text": audio_text
    }
    # #chat_message_history.add_ai_message("Hi")

    return jsonify(response_data)



def send_question_to_chatgpt(question, contextData):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    print(question)
    print(contextData)
    data = {
        
        "model": "gpt-3.5-turbo", #gpt-3.5-turbo
        # "prompt": question,
        "max_tokens": 150,
        "n": 1,
        "stop": None,
        "temperature": 0.8,
        "messages":[
                {"role": "system", "content": "You are a and knowledgebale AI agent that will assist a user in planning an agriculture operation."},
                {"role": "system", "content": "If the user asks for a topic unrelated to agriculture and/or climate change, kindly steer them towards the topic"},
                {"role": "system", "content": "If the user asks for a topic unrelated to agriculture and/or climate change and cannot be sigued into the topic, politely decline their request"},
                {"role": "system", "content": "The user may ask specific questions about permaculture. There is extra information in the context provided"},
                {"role": "system", "content": "The context from the assistant may provide the current location of the user (CurrentLocation), the beds of crops that the user is planning or tending to (GeoJSONPins), and various plants that the user has scouted in the environment (Biodiversity) "},
                {"role": "system", "content": "If the user asks about current actions on the farm, you can refer to the GeoJSONPins PlotPlans and BedNumber. The Vegetable will be listed, along with Start and endDates for each time"},
                {"role": "system", "content": "Please be specific whenever possible, naming the state or region and plants that are native to it can help as examples."},
                {"role": "system", "content": "The day number will be 0-365 (e.g. April 1 ist 91), the app does not calculate Leap Years, but you can "},
                {"role": "system", "content": "Actions taken on a farm may include planting (where the current day the one of first 3 days in the Plot Plans), harvesting if it's one of the last 5 days in the Plot Plans, tending to plants/helpful growing advice if the plant is growing, or bed maintence if nothing is growing in the bed."},
                {"role": "system", "content": "A user asking for advice may not have room for a lot of actions, so around 3 tips or action points/beds will be suitable"},
                {"role": "system", "content": "When answering questions related to biodiversity, focus on the importance of preserving and promoting the variety of species, habitats, and ecosystems in the environment. Provide suggestions on how the user can enhance biodiversity on their farm or in their local area, such as planting native species, creating habitats for wildlife, and implementing sustainable agricultural practices."},
                {"role": "assistant", "content": contextData},
                {"role": "user", "content": question}
        ]
    }

    response = requests.post(OPENAI_API_URL, headers=headers, json=data)
    response_data = response.json()
    print(response_data)
    if "choices" in response_data and len(response_data["choices"]) > 0:
        return response_data["choices"][0]["message"]["content"].strip()

    return None


def get_audio_text(chatgpt_response):
    # + "..."
    chatgpt_response = re.sub(r'[^\w\s?!.,]', '', chatgpt_response) + "..."
    escaped_query = requests.utils.quote(chatgpt_response)
    # response = requests.get(f"{AUDIO_SERVER_URL}?text={escaped_query}")
    # audio_text = response.text
    # print(audio_text)

    # return audio_text
    return f"{AUDIO_SERVER_URL}?text={escaped_query}"

#C:\GitRepos\FlaskDemo\modifiedSampleGraph.geojson
@app.route("/sampleGraph.geojson", methods=["GET"])
def getSampleGeoJSON():
    f = open("./modifiedSampleGraph.geojson",)
    response_data = json.load(f)
    f.close()

    return jsonify(response_data)

@app.route("/irrigation.geojson", methods=["GET"])
def getIrrigationJSON():
    f = open("./irrigation.geojson",)
    response_data = json.load(f)
    f.close()

    return jsonify(response_data)


@app.route("/biodiversity.geojson", methods=["GET"])
def getBioDiversityJSON():
    f = open("./biodiversity.geojson",)
    response_data = json.load(f)
    f.close()

    return jsonify(response_data)


@app.route('/biodiversitydemo/<path:path>')
def send_report(path):
    return send_from_directory('biodiversitydemo', path)

ConversationChains = {}


if __name__ == "__main__":
    # create_embeddings()
    # embedding_model = NVIDIAEmbeddings(model="nvolveqa_40k")
    # # Embed documents
    # embedding_path = "embed/"
    # docsearch = FAISS.load_local(folder_path=embedding_path, embeddings=embedding_model,
                                # allow_dangerous_deserialization=True)
    
    llm = ChatNVIDIA(model="llama2_70b")
    # ConversationChain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

    # chat = ChatNVIDIA(model="mixtral_8x7b", temperature=0.1, max_tokens=1000, top_p=1.0)

    # doc_chain = load_qa_chain(chat , chain_type="stuff", prompt=QA_PROMPT)

    # qa = ConversationalRetrievalChain(
        # retriever=docsearch.as_retriever(),
        # combine_docs_chain=doc_chain,
        # memory=memory,
        # question_generator=question_generator,
    # )
    
    # query = "What is Triton?"
    # result = qa({"question": query})
    # print(result.get("answer"))
    
    app.run(host="0.0.0.0", debug=True)

# if __name__ == "__main__":
    # app.run(debug=True)
