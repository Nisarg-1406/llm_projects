from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from core.state import State
import logging
import os
from pathlib import Path
from langchain.agents import AgentExecutor
# Set up logger
logger = logging.getLogger(__name__)

def agent_node(state: State, agent: AgentExecutor, name: str) -> State:
    """
    Process an agent's action and update the state accordingly.
    """
    logger.info(f"Processing agent: {name}")
    try:
        result = agent.invoke(state)
        logger.debug(f"Agent {name} result: {result}")
        
        output = result["output"] if isinstance(result, dict) and "output" in result else str(result)
        # print("State: ", state["hypothesis"])
        # print("OUTPUT: ", output)
        
        ai_message = AIMessage(content=output, name=name)
        # print("AI_message: ", ai_message)
        state["messages"].append(ai_message)
        state["sender"] = name
        
        if name == "hypothesis_agent" and not state["hypothesis"]: # not state["hypothesis"] means hypothesis not performed or user has asked to revise the hypothesis
            state["hypothesis"] = ai_message
            logger.info("Hypothesis updated")
        elif name == "process_agent":
            state["process_decision"] = ai_message
            logger.info("Process decision updated")
        elif name == "visualization_agent":
            state["visualization_state"] = ai_message
            logger.info("Visualization state updated")
        elif name == "searcher_agent":
            state["searcher_state"] = ai_message
            logger.info("Searcher state updated")
        elif name == "report_agent":
            state["report_section"] = ai_message
            logger.info("Report section updated")
        elif name == "quality_review_agent":
            state["quality_review"] = ai_message
            state["needs_revision"] = "revision needed" in output.lower()
            logger.info(f"Quality review updated. Needs revision: {state['needs_revision']}")
        
        logger.info(f"Agent {name} processing completed")
        return state
    except Exception as e:
        logger.error(f"Error occurred while processing agent {name}: {str(e)}", exc_info=True)
        error_message = AIMessage(content=f"Error: {str(e)}", name=name)
        return {"messages": [error_message]}

def human_choice_node(state: State) -> State:
    """
    Handle human input to choose the next step in the process.
    If regenerating hypothesis, prompt for specific areas to modify.
    """
    logger.info("Prompting for human choice")
    print("Please choose the next step:")
    print("1. Regenerate hypothesis")
    print("2. Continue the research process")
    
    while True:
        choice = input("Please enter your choice (1 or 2): ")
        if choice in ["1", "2"]:
            break
        logger.warning(f"Invalid input received: {choice}")
        print("Invalid input, please try again.")
    
    if choice == "1":
        modification_areas = input("Please specify which parts of the hypothesis you want to modify: ")
        content = f"Regenerate hypothesis. Areas to modify: {modification_areas}"
        state["hypothesis"] = ""
        logger.info("Hypothesis cleared for regeneration")
        logger.info(f"Areas to modify: {modification_areas}")
    else:
        content = "Continue the research process"
        state["process"] = "Continue the research process"
        logger.info("Continuing research process")
    
    human_message = HumanMessage(content=content)
    
    state["messages"].append(human_message)
    state["sender"] = 'human'
    
    logger.info("Human choice processed")
    return state

def human_review_node(state: State) -> State:
    """
    Display current state to the user and update the state based on user input.
    Includes error handling for robustness.
    """
    try:
        print("Current research progress:")
        print(state)
        print("\nDo you need additional analysis or modifications?")
        
        while True:
            user_input = input("Enter 'yes' to continue analysis, or 'no' to end the research: ").lower()
            if user_input in ['yes', 'no']:
                break
            print("Invalid input. Please enter 'yes' or 'no'.")
        
        if user_input == 'yes':
            while True:
                additional_request = input("Please enter your additional analysis request: ").strip()
                if additional_request:
                    state["messages"].append(HumanMessage(content=additional_request))
                    state["needs_revision"] = True
                    break
                print("Request cannot be empty. Please try again.")
        else:
            state["needs_revision"] = False
        
        state["sender"] = "human"
        logger.info("Human review completed successfully.")
        return state
    
    except KeyboardInterrupt:
        logger.warning("Human review interrupted by user.")
        return None
    
    except Exception as e:
        logger.error(f"An error occurred during human review: {str(e)}", exc_info=True)
        return None
    
def refiner_node(state: State, agent: AgentExecutor, name: str) -> State:
    """
    Read MD file contents and PNG file names from the specified storage path,
    add them as report materials to a new message,
    then process with the agent and update the original state.
    If token limit is exceeded, use only MD file names instead of full content.
    """
    try:
        storage_path = Path(os.getenv('STORAGE_PATH', './data_storage/'))
        
        materials = []
        md_files = list(storage_path.glob("*.md"))
        png_files = list(storage_path.glob("*.png"))
        
        for md_file in md_files:
            with open(md_file, "r", encoding="utf-8") as f:
                materials.append(f"MD file '{md_file.name}':\n{f.read()}")
        
        materials.extend(f"PNG file: '{png_file.name}'" for png_file in png_files)
        
        combined_materials = "\n\n".join(materials)
        report_content = f"Report materials:\n{combined_materials}"
        
        refiner_state = state.copy()
        refiner_state["messages"] = [BaseMessage(content=report_content)]
        
        try:
            result = agent.invoke(refiner_state)
        except Exception as token_error:
            # If token limit is exceeded, retry with only MD file names
            logger.warning("Token limit exceeded. Retrying with MD file names only.")
            md_file_names = [f"MD file: '{md_file.name}'" for md_file in md_files]
            simplified_materials = "\n".join(md_file_names)
            simplified_report_content = f"Report materials (file names only):\n{simplified_materials}"
            
            refiner_state["messages"] = [BaseMessage(content=simplified_report_content)]
            result = agent.invoke(refiner_state)
        
        # Update original state
        state["messages"].append(AIMessage(content=result))
        state["sender"] = name
        
        logger.info("Refiner node processing completed")
        return state
    except Exception as e:
        logger.error(f"Error occurred while processing refiner node: {str(e)}", exc_info=True)
        state["messages"].append(AIMessage(content=f"Error: {str(e)}", name=name))
        return state
    
logger.info("Agent processing module initialized")