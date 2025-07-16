from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import os
from logger import setup_logger

logger = setup_logger()

@tool
def list_directory_contents(directory: str = './data_storage/') -> str:
    """
    List the contents of the specified directory.
    
    Args:
        directory (str): The path to the directory to list. Defaults to the data storage directory.
    
    Returns:
        str: A string representation of the directory contents.
    """
    try:
        logger.info(f"Listing contents of directory: {directory}")
        contents = os.listdir(directory)
        logger.debug(f"Directory contents: {contents}")
        return f"Directory contents :\n" + "\n".join(contents)
    except Exception as e:
        logger.error(f"Error listing directory contents: {str(e)}")
        return f"Error listing directory contents: {str(e)}"

def create_agent(
    llm: ChatOpenAI,
    tools: list[tool],
    system_message: str,
    team_members: list[str],
    working_directory: str = './data_storage/'
) -> AgentExecutor:
    """
    Create an agent with the given language model, tools, system message, and team members.
    
    Parameters:
        llm (ChatOpenAI): The language model to use for the agent.
        tools (list[tool]): A list of tools the agent can use.
        system_message (str): A message defining the agent's role and tasks.
        team_members (list[str]): A list of team member roles for collaboration.
        working_directory (str): The directory where the agent's data will be stored.
        
    Returns:
        AgentExecutor: An executor that manages the agent's task execution.

    Flow:
    - Ensures the ListDirectoryContents tool is available for the agent.
    - Generates a system prompt describing:
        - The agent’s role and responsibilities.
        - Available tools and team members.
        - The initial contents of the working directory.
    - Creates a structured prompt template with placeholders for different AI-generated responses (e.g., hypothesis, process, visualization state).
    - Builds the agent using create_openai_functions_agent() and the given tools.

    - MessagesPlaceholder(variable_name="messages") - Acts as a dynamic placeholder where past conversation history (messages) can be inserted. This allows the model to maintain context in an interactive conversation.
    - ("ai", "hypothesis: {hypothesis}") 
        - Defines a specific message format where the AI generates a "hypothesis". 
        - {hypothesis} is a placeholder that gets filled dynamically when running the agent.
        - Example: prompt.format(hypothesis="Based on the data, the trend indicates an increase."). The AI will recieve as -> ai: hypothesis: Based on the data, the trend indicates an increase.

    """
    
    logger.info("Creating agent")

    # Ensure the ListDirectoryContents tool is available
    if list_directory_contents not in tools:
        tools.append(list_directory_contents)

    # Prepare the tool names and team members for the system prompt
    tool_names = ", ".join([tool.name for tool in tools])
    team_members_str = ", ".join(team_members)

    # List the initial contents of the working directory
    initial_directory_contents = list_directory_contents(working_directory)

    # Create the system prompt for the agent
    system_prompt = (
        "You are a specialized AI assistant in a data analysis team. "
        "Your role is to complete specific tasks in the research process. "
        "Use the provided tools to make progress on your task. "
        "If you can't fully complete a task, explain what you've done and what's needed next. "
        "Always aim for accurate and clear outputs. "
        f"You have access to the following tools: {tool_names}. "
        f"Your specific role: {system_message}\n"
        "Work autonomously according to your specialty, using the tools available to you. "
        "Do not ask for clarification. "
        "Your other team members (and other teams) will collaborate with you based on their specialties. "
        f"You are chosen for a reason! You are one of the following team members: {team_members_str}.\n"
        f"The initial contents of your working directory are:\n{initial_directory_contents}\n"
        "Use the ListDirectoryContents tool to check for updates in the directory contents when needed."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("ai", "hypothesis: {hypothesis}"),
        ("ai", "process: {process}"),
        ("ai", "process_decision: {process_decision}"),
        ("ai", "visualization_state: {visualization_state}"),
        ("ai", "searcher_state: {searcher_state}"),
        ("ai", "code_state: {code_state}"),
        ("ai", "report_section: {report_section}"),
        ("ai", "quality_review: {quality_review}"),
        ("ai", "needs_revision: {needs_revision}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    
    logger.info("Agent created successfully")
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False)

def create_supervisor(llm: ChatOpenAI, system_prompt: str, members: list[str]) -> AgentExecutor:
    """
    Purpose: FOR ROUTING to which agent should perform the next task

    In function_def: 
    - "next" → Specifies the next agent (or "FINISH").
    - "task" → Describes the task to be performed by the selected agent.

    PromptTemplate: 
    - "system_prompt" → Provides instructions on how the supervisor should behave.
    - MessagesPlaceholder("messages") → Stores the conversation history dynamically.
    - The decision prompt asks:
        - Who should act next? (Choose from options.)
        - What task should they perform? (Assign a relevant task.)
    - .partial(...) → Pre-fills {options} and {team_members} placeholders.
    """
    logger.info("Creating supervisor")
    
    options = ["FINISH"] + members
    
    function_def = {
        "name": "route",
        "description": "Select the next role and assign a task.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
                "task": {
                    "title": "Task",
                    "type": "string",
                    "description": "The task to be performed by the selected agent"
                }
            },
            "required": ["next", "task"],
        },
    }
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next? "
                "Or should we FINISH? Select one of: {options}. "
                "Additionally, specify the task that the selected role should perform."
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    
    logger.info("Supervisor created successfully")
    
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )