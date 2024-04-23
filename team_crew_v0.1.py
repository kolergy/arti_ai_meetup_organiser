import os
from   crewai           import Agent, Task, Crew, Process
from   crewai_tools     import BaseTool, WebsiteSearchTool
from   langchain_groq   import ChatGroq
from   langchain_openai import ChatOpenAI
from   langchain.llms   import Ollama
from   dotenv           import load_dotenv

load_dotenv()

 # load the Ai news
AI_news_file_path               = '/home/me/Cozy Drive/AI_meetup/AI_news_of_the_month.txt'
with open(AI_news_file_path, 'r') as file:
    AI_news_text = file.read()

participants_requests_file_path = '/home/me/Cozy Drive/AI_meetup/participants_requests.txt'
with open(participants_requests_file_path, 'r') as file:
    participants_requests_text = file.read()

Ollama_llama3 = Ollama(model="llama3")

#openai_api_key                     = os.getenv('OPENAI_API_KEY')
#llm_openai = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", max_retries=2)
                                               

#groq_api_key                        = os.getenv('GROQ_API_KEY')
#llm_groq = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192", max_retries=2)

#current_llm = llm_groq
#current_llm = llm_openai
current_llm = Ollama_llama3

arti_ai_specialist = Agent(
    role             = 'AI topic identification expert',
    goal             = 'Identify the relevant ai topics and Produce a list of topics ordered by relevance to the meetup participants.',
    verbose          = True,
    memory           = True,
    backstory        = (
        "you are an AI wizard With a deep background in AI research and development, you are the go-to "
        "expert for all things AI-related. You have a knack for understanding complex AI concepts and link them to real-world applications."
        "You are the one who will be able to understand the participant requests and the AI news and link them together when it makes sense."
        "as the event takes place in a Fablab there is an emphasis on the practical side of AI an the open-source / self hosting aspect of it."
    ),
    #tools            = [WebsiteSearchTool()],  
    tools            = [],  
    llm              = current_llm,
    allow_delegation = False,
    Max_iter         = 10
)

arti_event_planner = Agent(
    role             = 'AI Event Planner',
    goal             = 'Select three subject to be developped dutring the meetup event',
    verbose          = True,
    memory           = True,
    backstory        = (
        "Expert in curating AI-focused events, this agent uses data-driven insights "
        "to select impactful topics and structure them into compelling agendas."
    ),
    tools            = [], 
    llm              = current_llm,
    allow_delegation = True,
    Max_iter         = 10
)


# Task to select key topics
Analyze_topics = Task(
    description      = (
        "**Task:**: Generate a potential list of topics for the AI meetup."
        "**Description:**:generate a list of topics based on participant interests and current AI news."
        " - Analyze the participant requests in light of the AI news "
        " - Thinking step by step identify the links between them all."
        " - Return a list of topics ordered by relevance to the participants and importance."
        "**Parameters:**:"
        f" - AI news of the month in triple ticks '''{AI_news_text}''' "
        f" - Participants requests in triple ticks '''{participants_requests_text}''' "
    ),
    tools            = [],
    expected_output  = 'A curated list of topics ordered by relevance to the participants and importance .',
    agent            = arti_ai_specialist,
    allow_delegation = True,
)

# Task to generate the meetup agenda
select_tasks = Task(
    description      = (
        "**Task:**: select three topics for the AI meetup."
        "**Description:**: Identify the three engaging topics most relevant to the participants requests."
        "and ensure that:"
        "  - the three topics are consistants with the participants requests."
        "  - the three topics are within the fablab's spirit."
        "  - if possible the topics are in the AI news of the month."
        "  - there are only three topics."
        "for each topic selected. provide a rationale, a title, and a brief description."
        "**Parameters:**:"
        f" - AI topics in triple ticks '''{Analyze_topics.output}''' "
        f" - AI news of the month in triple ticks '''{AI_news_text}''' "
        f" - Participants requests in triple ticks '''{participants_requests_text}''' "
        "**Output:**: A json file with the three selected topics. containing the following keys:"
        "  - rationale: the reason why the topic was selected"
        "  - topic_title: the title of the topic selected"
        "  - topic_description: a brief description of the topic"

    ),
    expected_output  = 'A structured agenda for the AI meetup.',
    Context          = [Analyze_topics],
    tools            = [],
    agent            = arti_event_planner,
    allow_delegation = False
)


# Forming the AI Event Planning crew
# Crew setup
ai_meetup_crew = Crew(
    agents      = [arti_ai_specialist, arti_event_planner],
    tasks       = [Analyze_topics    , select_tasks      ],
    #max_rpm     = 2, # Maximum Rounds Per Minute GROQ API limit is 30 but hitting TPM
    verbose     = True,
    process     = Process.sequential,
    #full_output = True,
)

# Execute the crew with input files
result = ai_meetup_crew.kickoff()
print(result)