from llm import llm
from graph import graph
from utils import get_session_id
from tools.vector import get_chunk_text
from tools.cypher import cypher_qa

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory


# Create a movie chat chain
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a particle physics expert providing information about particle physics analyses."),
        ("human", "{input}"),
    ]
)

physics_chat = chat_prompt | llm | StrOutputParser()

# Create a set of tools
tools = [
    Tool.from_function(
        name="General Physics Chat",
        description="For generic particle physics chat not covered by other tools",
        func=physics_chat.invoke,
    ),
    Tool.from_function(
        name="Paper Text Search",
        description="""For when you need to find information about vector boson fusion (VBF) analysis based on specific questions about the analysis.
        This might include information like how events are selected in different analysis regions, what are the uncertainties being considered,
        information about the particle detector being used, or information about triggers being used.""",
        func=get_chunk_text,
    ),
]

# Create chat history callback
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# Create the agent
# agent_prompt = hub.pull("hwchase17/react-chat")

agent_prompt = PromptTemplate.from_template("""
You are an expert providing information about particle physics.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to particle physics.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Create a handler to call the agent
def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI.
    """
    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},
    )

    return response["output"]