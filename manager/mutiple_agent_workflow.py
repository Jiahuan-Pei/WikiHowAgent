import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

# Import the Teacher and Learner agents
from agents.teacher_agent import TeacherLLMAgent
from agents.learner_agent import LearnerLLMAgent
from agents.evaluator_agent import ConversationEvaluator

from concurrent.futures import ThreadPoolExecutor
from utils.util import logger
from config import args, config_teacher, config_learner, config_evaluator

# Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize global variables for the agents
teacher_agent = None
learner_agent = None

# Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Conversation Nodes
def teacher_node(state):
    messages = state["messages"]
    teacher_response = teacher_agent.run(messages[-1].content if messages else "")
    return {"messages": [AIMessage(content=teacher_response)]}

def learner_node(state):
    messages = state["messages"]
    learner_response = learner_agent.run(messages[-1].content)
    return {"messages": [HumanMessage(content=learner_response)]}

# Conversation Nodes (Batch Versions)
def teacher_node_batch(state, batch_size=2):
    messages = state["messages"]
    batch_utterances = [msg.content for msg in messages[-batch_size:]]
    teacher_responses = teacher_agent.run_batch(batch_utterances)
    return {"messages": [AIMessage(content=resp) for resp in teacher_responses]}

def learner_node_batch(state, batch_size=2):
    messages = state["messages"]
    batch_instructions = [msg.content for msg in messages[-batch_size:]]
    learner_responses = learner_agent.run_batch(batch_instructions)
    return {"messages": [HumanMessage(content=resp) for resp in learner_responses]}

# Flow Control: Determines Conversation Path
def should_continue(state, max_steps=15):
    messages = state["messages"]

    if len(messages) >= max_steps or any("FINISHED" in m.content for m in messages):
        return "end"

    last_message = messages[-1].content.lower()

    if isinstance(messages[-1], HumanMessage) and "FINISHED" not in last_message:
        return "teacher"
    elif isinstance(messages[-1], AIMessage) and "FINISHED" not in last_message:
        return "learner"

    return "end"

# Generate Conversation
def generate_single_conversation(tutorial_document, config_teacher, config_learner):
    global teacher_agent, learner_agent

    teacher_agent = TeacherLLMAgent(config_teacher, tutorial_document=tutorial_document)
    learner_agent = LearnerLLMAgent(config_learner)

    graph_builder = StateGraph(State)
    graph_builder.add_node("learner", learner_node)
    graph_builder.add_node("teacher", teacher_node)
    graph_builder.add_edge("teacher", "learner")
    graph_builder.add_conditional_edges("learner", should_continue, {"end": END, "teacher": "teacher"})
    graph_builder.add_edge(START, "teacher")

    simulation = graph_builder.compile()
    conversation = []
    state = {"messages": []}

    for chunk in simulation.stream(state):
        if END not in chunk:
            messages = chunk["teacher"] if 'teacher' in chunk else chunk["learner"]
            for message in messages['messages']:
                role = "Teacher" if isinstance(message, AIMessage) else "Learner"
                conversation.append(f"{role}: {message.content.replace('\n', '')}")
    return conversation

# Processing a tutorial document
def process_method(method, method_id, summary, source_tutorial_path, methods=None):
    if methods is None:
        methods = []

    if method:
        tutorial_document = [f"{summary} {method['title']}"] + method['steps']
    else:
        tutorial_document = [summary] + [s['title'] + ". " + s['steps'][0] for s in methods if s['steps']]

    # If tutorial has no steps after processing, skip it
    if len(tutorial_document) <= 1:  
        logger.warning(f"Skipping tutorial with no valid steps for method {method_id} : {source_tutorial_path}")
        return None 

    # Generate and evaluate one conversation
    conversation = generate_single_conversation(tutorial_document=tutorial_document, config_teacher=config_teacher, config_learner=config_learner)
    evaluation_results = ConversationEvaluator(config_evaluator).evaluate(conversation, tutorial_document)  

    logger.info("\n--- Full Conversation ---\n" + "\n".join(conversation))
    # logger.info(evaluation_results)
    return {
        'id': str(uuid.uuid4()),
        'source_tutorial_path': source_tutorial_path,
        'method_id': method_id,
        'tutorial': tutorial_document,
        'conversation': conversation,
        'evaluation': evaluation_results
    }

# Batch Processing for Methods: Runs multiple tutorial methods in parallel.
def process_method_batch(methods_batch, summary, source_tutorial_path):
    """Process multiple methods in batch using threading."""
    results = []
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = [executor.submit(process_method, method, i + 1, summary, source_tutorial_path) 
                   for i, method in enumerate(methods_batch)]
        results = [f.result() for f in futures if f.result()]
    return results
