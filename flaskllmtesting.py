import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA 
from langchain_nvidia_ai_endpoints.embeddings import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, BaseMessage

import os, re, json
import requests

### Statefully manage chat history ###
store = {}

class ConversationBufferWindowMemory(BaseChatMessageHistory):
    """A chat message history that only keeps the last K messages."""
    
    def __init__(self, buffer_size: int):
        super().__init__()
        self.buffer_size = buffer_size
        self.messages: List[BaseMessage] = []

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add a list of messages to the store, keeping only the last K."""
        self.messages.extend(messages)
        # Ensure only the last K messages are kept
        self.messages = self.messages[-self.buffer_size:]


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = SQLChatMessageHistory(
            session_id = session_id,
            connection_string = "sqlite:///ChatMessageHistory.db"
        ) #ChatMessageHistory()
    return store[session_id]

#This will get a url that can be passed to Mozilla TTS
#Should switch over to RIVA when NIM becomes available
def get_audio_text(chatgpt_response):
    # + "..."
    chatgpt_response = re.sub(r'[^\w\s?!.,]', '', chatgpt_response) + "..."
    escaped_query = requests.utils.quote(chatgpt_response)
    # response = requests.get(f"{AUDIO_SERVER_URL}?text={escaped_query}")
    # audio_text = response.text
    # print(audio_text)

    # return audio_text
    return f"{AUDIO_SERVER_URL}?text={escaped_query}"

#############################
### SECTION: Setting Keys ###
#############################

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY","sk")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

nvapi_key = os.environ.get("NVIDIA_API_KEY","nvapi") #getpass.getpass("Enter your NVIDIA API key: ")
os.environ["NVIDIA_API_KEY"] = nvapi_key
assert nvapi_key.startswith("nvapi-"), f"{nvapi_key}... is not a valid key"
DeploymentIP = "127.0.0.1"
AUDIO_SERVER_URL = "http://"+DeploymentIP+":5002/api/tts"
   
# Using ChatOpenAI for langchain.
# There were 400 issues on using message history,
# So using Nvidia NIM endpoints will use requests . 
# Formulating the request like in AvatarResponseFlask.py works fine, 
# though message history needs manual addition and culling .
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0,)

multiModal = ChatOpenAI( 
    model="gpt-4",  
    temperature=0,
)

### Construct retriever ###
### This is a sample retriever, but could be fitted to scrape pertinent journals
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs =  loader.load()
#print(docs)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


# This part of the chain may be causing issues with uploading images. 
# Need to research this to confirm
##############################
### Contextualize question ###
##############################
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

#####################################
### Main System Prompt and Chains ###
#####################################
if True:
    MainSystemPrompt = "You are a friendly and knowledgebale AI agent that will assist a user in planning an agriculture operation."+\
    "If the user asks for a topic unrelated to agriculture and/or climate change, kindly steer them towards the topic"+\
    "If the user asks for a topic unrelated to agriculture and/or climate change and cannot be sigued into the topic, politely decline their request"+\
    "The user may ask specific questions about permaculture. There is extra information in the context provided"+\
    "The context from the assistant may provide the current location of the user (CurrentLocation), the beds of crops that the user is planning or tending to (GeoJSONPins), and various plants that the user has scouted in the environment (Biodiversity) "+\
    "If the user asks about current actions on the farm, you can refer to the GeoJSONPins PlotPlans and BedNumber. The Vegetable will be listed, along with Start and endDates for each time"+\
    "Please be specific whenever possible, naming the state or region and plants that are native to it can help as examples."+\
    "The day number will be 0-365 (e.g. April 1 ist 91), the app does not calculate Leap Years, but you can "+\
    "Actions taken on a farm may include planting (where the current day the one of first 3 days in the Plot Plans), harvesting if it's one of the last 5 days in the Plot Plans, tending to plants/helpful growing advice if the plant is growing, or bed maintence if nothing is growing in the bed."+\
    "A user asking for advice may not have room for a lot of actions, so around 3 tips or action points/beds will be suitable"+\
    "When answering questions related to biodiversity, focus on the importance of preserving and promoting the variety of species, habitats, and ecosystems in the environment."+\
    "Keep the responses concise, but please be thorough in your answer."+\
    "\n\n{context}"

    MainChat_prompt = ChatPromptTemplate.from_messages(
        [
            #("system", system_prompt),
            ("system", MainSystemPrompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    MainChat_question_answer_chain = create_stuff_documents_chain(llm, MainChat_prompt)

    MainChat_rag_chain = create_retrieval_chain(history_aware_retriever, MainChat_question_answer_chain)

    ########################################################
    ### Multi Modal version of chain for image searching ###
    ########################################################
    MainChat_mm_question_answer_chain = create_stuff_documents_chain(multiModal, MainChat_prompt)

    MainChat_mm_rag_chain = create_retrieval_chain(history_aware_retriever, MainChat_mm_question_answer_chain)


############################################
### Question Generator Prompt and Chains ###
############################################
if True:
    QuestionGeneratorPrompt = """
    You are a AI subsystem that will create question recommendations to ask other AI agents.
    Using the supplied context, you'll create 5 short questions that will lead to further conversation.
    The questions will be in a numbered list, separated into lines.
    The first three questions will be General questions (e.g. "What are the alernatives to planting X?", "What are the common mistakes for Y to avoid?", etc)
    The fourth question should ask for any scientific studies relating to the most recent prompts (e.g. "Is there science on planting X in Y month?", "What is the research on using X,Y, and Z fertilizers on this plot?") 
    The fifth question should be related to buying something helpful from recent past conversation (e.g. "Can you send a link to by X seeds?", "What tractors are available near me?"  
    Below is the supplied context

    \n\n{context}

    """
    
    # To prevent adding the question generator to the main history;
    # We also add the message history of the main conversation separately
    QuestionGenerator_prompt = ChatPromptTemplate.from_messages(
        [
            #("system", system_prompt),
            ("system", QuestionGeneratorPrompt),
            MessagesPlaceholder("chat_history"),
            #MessagesPlaceholder("message_history"),
            ("system", "{message_history}"),
            ("human", "{input}"),
        ]
    )
    QuestionGenerator_question_answer_chain = create_stuff_documents_chain(llm, QuestionGenerator_prompt)

    QuestionGenerator_rag_chain = create_retrieval_chain(history_aware_retriever, QuestionGenerator_question_answer_chain)


# Getting session histories for debugging
ConversationalQs = get_session_history("61224803")
GenerativeQs = get_session_history("GeneratingQuestionsChat")

#########################################
### Creating Runnables for invocation ###
#########################################
if True:
    conversational_MainChat_rag_chain = RunnableWithMessageHistory(
        MainChat_rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    conversational_MainChat_mm_rag_chain = RunnableWithMessageHistory(
        MainChat_mm_rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    conversational_QuestionGenerator_rag_chain = RunnableWithMessageHistory(
        QuestionGenerator_rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

            

##########################
### Start of Flask App ###
##########################
from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
import asyncio, time 

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with your own secret key

socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/GetMessageHistory', methods=[ "POST"])
def GetMessageHistory():
    if request.method == "POST":
        messageList = []
        for message in get_session_history("MainChat").get_messages():
            messageList.append({
                "type":message.type,
                "content":message.content,
            })
        return {"message":messageList}

def sendConversationalMessage(userMessage = "", userImage = None):
    sessionID = "MainChat"
    
    newHumanMessage = HumanMessage(content=[
        {"type": "text", "text": userMessage},
    ])
    # if len(userImage)>512:
        # sessionID = "TestingImages"
        # newHumanMessage.content.append({
            # "type": "image_url", 
            # "image_url": {
                # "url": f"data:image/png;base64,{userImage}"
            # }
        # })
        # conversationRagResponse = conversational_MainChat_mm_rag_chain.invoke(
            # {"input": newHumanMessage},
            # config={
                # "configurable": {"session_id": sessionID}
                # #"configurable": {"session_id": "MainChat"}
            # },  # constructs a key "abc123" in `store`.
        # )

    # else:
    conversationRagResponse = conversational_MainChat_rag_chain.invoke(
        {"input": userMessage},
        config={
            "configurable": {"session_id": sessionID}
            #"configurable": {"session_id": "MainChat"}
        },  # constructs a key "abc123" in `store`.
    )
    socketio.emit('LLMBroadcast', conversationRagResponse["answer"])
    
    
@app.route('/NewUserMessage', methods=[ "POST"])
def NewUserMessage():
    if request.method == "POST":
        #print() 
        sendConversationalMessage( request.json["message"], request.json.get("image"))
        
        return request.json

def sendQuestionRecommendations(userMessage = ""):
    thatHistory = get_session_history("questionAndAnswer")
    thatHistory.clear()
    questionGeneratorRagResponse = conversational_QuestionGenerator_rag_chain.invoke(
        {
            "input": userMessage,
            "message_history": str(get_session_history("MainChat"))[-5000:]
        },
        config={
            "configurable": {"session_id": "questionAndAnswer"}
        },  # constructs a key "abc123" in `store`.
    )
    #print(questionGeneratorRagResponse["answer"])
    formattedQuestions = []
    for line in questionGeneratorRagResponse["answer"].split("\n"):
        if(line and line[0].isdigit()):
            formattedQuestions.append(line[2:])
        
    socketio.emit('QuestionBroadcast', {"questions": formattedQuestions})
    

@app.route('/GetQuestionRecommendations', methods=[ "POST"])
def GetQuestionRecommendations():
    if request.method == "POST":
        #print() 
        sendQuestionRecommendations( request.json["message"])
        
        return request.json



@app.route('/files/<path:path>')
def send_file(path):
    return send_from_directory('files', path)

@app.route('/SaveUserMap', methods=['POST'])
def SaveUserMap():
    data = request.get_json()
    path = 'files/usermap.json'
    with open(path, 'w') as f:
        json.dump(data, f)
    return {"status":"success"}


    

@app.route('/GetAIAudio', methods=[ "POST"])
def GetAIAudio():
    if request.method == "POST":
        #print() 
        audio_text = get_audio_text(request.json["message"].replace('\n',''))
        return {
            "audioURL":audio_text
        }
    
    
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('message')
def handle_message(data):
    print('Received message:', data)
    socketio.emit('response', 'Server received your message: ' + data)


if __name__ == "__main__":
    socketio.run(app, debug=True)
    
#OK!