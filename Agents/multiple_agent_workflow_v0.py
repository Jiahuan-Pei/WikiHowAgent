import os
import yaml
import glob
import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from Agents.util import setup_llm_and_embeddings, read_all_file_suffix_X  # Assuming this is defined in utils.py
from auto_evaluator import ConversationEvaluator
import uuid
from tqdm import tqdm
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.setup_logger import *
import numpy as np

# Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Teacher Agent
class TeacherLLMAgent:
    def __init__(self, config_file, tutorial_document=None):
        self.tutorial_document = tutorial_document
        self.current_step = 1
        self.total_steps = len(tutorial_document) - 1 # The 0th restores the summary
        self.llm, _, self.config = setup_llm_and_embeddings(config_file)

        self.prompt_template = PromptTemplate(
            input_variables=["current_step", "instruction", "user_utterance"],
            template=(
                "You are a teacher guiding a learner through a tutorial. "
                "Your job is to give instructions for the current step and answer any questions."
                "Summary: {summary}\n"
                "Step instruction: {instruction}\n"
                "Learner: {user_utterance}\n"
                "Teacher:"
            ),
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=True)

    def run(self, user_utterance: str) -> str:
        """Process the learner's response and generate the teacher's reply."""
        if self.current_step >= self.total_steps:
            return "FINISHED"

        summary = self.tutorial_document[0] # Assuming the first element is the summary of the tutorial
        instruction = self.tutorial_document[self.current_step]

        # Generate teacher response
        teacher_response = self.chain.run(
            summary=summary,
            current_step=self.current_step + 1,
            instruction=instruction,
            user_utterance=user_utterance
        )

        # Check if learner wants to proceed
        next_step_decision = self.llm.invoke(
            f"Based on this response: '{user_utterance}', does the learner understand and want to proceed? "
            f"Reply with 'yes' or 'no'."
        ).content.lower()

        if "yes" in next_step_decision:
            self.current_step += 1  # Move to next step

        if self.current_step >= self.total_steps:
            return f"Step FINAL: {teacher_response}\nThe tutorial is now complete. FINISHED."

        return teacher_response

# Learner Agent
class LearnerLLMAgent:
    def __init__(self, config_file):
        self.llm, _, self.config = setup_llm_and_embeddings(config_file)

        self.prompt_template = PromptTemplate(
            input_variables=["instruction"],
            template=(
                "You are a student learning from a tutorial. "
                "Read the teacher's instruction and respond naturally. "
                "If the step is clear, acknowledge it shortly in a conversational way. "
                "If unclear, ask a brief and specific question about what is confusing."
                "If the teacher mentions 'FINISHED' or acknowledges the completion of the tutorial, respond by briefly thanking the teacher. Do not ask questions.\n\n"
                "Teacher: {instruction}\n"
                "Learner:"
            ),
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=True)

    def run(self, instruction: str) -> str:
        return self.chain.run(instruction=instruction)

# Nodes for Conversation
def teacher_node(state):
    messages = state["messages"]
    teacher_response = teacher_agent.run(messages[-1].content if messages else "")
    return {"messages": [AIMessage(content=teacher_response)]}

def learner_node(state):
    messages = state["messages"]
    learner_response = learner_agent.run(messages[-1].content)
    return {"messages": [HumanMessage(content=learner_response)]}

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


def generate_single_conversation(tutorial_document, logger):
    """Generate a conversation between teacher and learner agents based on a tutorial document.
    
    Args:
        tutorial_document: A list containing tutorial steps where:
            - First element is the summary
            - Subsequent elements are individual steps
            
    Example:
        tutorial_document = [
            "Summary: an introduction of the whole tutorial",
            "Fill a container with salt.",
            "Squeeze a little tempera paint into the salt.",
            "Mix with a spoon until evenly distributed.",
            "Let it dry overnight.",
            "Test before using in crafts."
        ]
        conversation = generate_single_conversation(tutorial_document)
    """
    global teacher_agent, learner_agent

    # Initialize Agents
    teacher_agent = TeacherLLMAgent(config_file, tutorial_document=tutorial_document)
    learner_agent = LearnerLLMAgent(config_file)

    # Define Graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("learner", learner_node)
    graph_builder.add_node("teacher", teacher_node)

    graph_builder.add_edge("teacher", "learner")
    graph_builder.add_conditional_edges("learner", should_continue, {"end": END, "teacher": "teacher"})
    graph_builder.add_edge(START, "teacher")

    # Compile Simulation
    simulation = graph_builder.compile()

    # Run simulation and store the conversation
    conversation = []
    state = {"messages": []}

    for chunk in simulation.stream(state):
        if END not in chunk:
            messages = chunk["teacher"] if 'teacher' in chunk else chunk["learner"]
            for message in messages['messages']:
                role = "Teacher" if isinstance(message, AIMessage) else "Learner"
                conversation.append(f"{role}: {message.content.replace('\n', '')}")

    # Print the full conversation
    # logger.info("This is an INFO message.")
    logger.info("\n--- Full Conversation ---\n" + "\n".join(conversation))
    return conversation

def process_method(method, method_id, summary, source_tutorial_path, methods=None, logger=None):
    """Helper function to process a single method and generate its dialog."""
    if methods is None:
        methods = []  # Default to an empty list if methods is not provided

    # Build tutorial_document
    if method:
        # For multiple methods: combine summary + title, and steps
        tutorial_document = [f"{summary} {method['title']}"] + method['steps']
    else:
        # For single method: combine summary, and steps
        tutorial_document = [summary] + [s['title'] + ". " + s['steps'][0] for s in methods if s['steps']]

    # Generate conversation and evaluate
    conversation = generate_single_conversation(tutorial_document=tutorial_document, logger=logger)
    evaluation_results = ConversationEvaluator(logger=logger).evaluate(conversation, tutorial_document)
    return {
        'id': str(uuid.uuid4()),
        'source_tutorial_path': source_tutorial_path,
        'method_id': method_id,
        'tutorial': tutorial_document,
        'conversation': conversation,
        'evaluation': evaluation_results
    }

def main():
    count_conversation = 0
    docs = read_all_file_suffix_X(mdir=mdir, suffix='.json') # , max_doc=2 # Doc = 3680

    total_dialogs = []
    total_scores = {metric: [] for metric in EVAL_METRICS}
    for doc in tqdm(docs):
        source_tutorial_path = doc['path']
        target_dialogue_path = source_tutorial_path.replace('.json', '.txt')
        log_path = source_tutorial_path.replace('.json', '.log')
        # Setup logger for
        logger = setup_logger(log_file=log_path, log_level=logging.INFO)
        dialogs = []
        if glob.glob(target_dialogue_path):
            logger.info(f'Skip: {source_tutorial_path} as it is processed before.')
            continue
        else:
            logger.info(f'##### {count_conversation}')
            summary = doc['introduction']
            methods = doc['methods']
            # Multiple methods if title contains 'Method x of Y:'
            if re.match(r'Method \d+ of \d+:.*', methods[0]['title']):
                for i, method in enumerate(methods):
                    count_conversation += 1
                    dialog = process_method(method, i + 1, summary + method['title'], source_tutorial_path, None, logger=logger)
                    dialogs.append(dialog)
                    for metric in EVAL_METRICS:
                        if metric in dialog['evaluation']:
                            total_scores[metric].append(dialog['evaluation'][metric])                   
                    
            # One method only
            else:
                count_conversation += 1
                dialog = process_method(None, 1, summary, source_tutorial_path, methods, logger=logger)
                dialogs.append(dialog)
                for metric in EVAL_METRICS:
                    if metric in dialog['evaluation']:
                        total_scores[metric].append(dialog['evaluation'][metric])               
        total_dialogs.extend(dialogs)
        # Save dialogs to a JSON file in the same folder for each doc
        with open(target_dialogue_path, 'w') as f:
            json.dump(dialogs, f, indent=4)
    
    # Compute average scores
    avg_scores = {metric: (np.mean(values), np.std(values)) for metric, values in total_scores.items() if values}

    # Save all dialogs together
    dialouge_json = {
        'config_file': config_file,
        'total_evaluation': avg_scores,
        'num_conversation': count_conversation,
        'total_conversations': total_dialogs
    }        
    with open(config_file.replace('conf', 'result').replace('yaml', 'json'), 'w') as f:
        json.dump(dialouge_json, f, indent=4)    
                        
    logger.info(f'Generate {count_conversation} conversations!!')
    # logger.info(dialouge_json)


if __name__ == "__main__":
    global mdir, config_file
    mdir='./data/wikihow'
    config_file = 'conf/ollma-llama3.yaml' 
    EVAL_METRICS = ["Question Ratio", "Completion Achieved", "Diversity Score", 
                    "Engagement", "Coherence", "Depth", "Relevance", "Progress", "Naturalness", "Truthfulness",
                    "BLEU", "METEOR", "BERTScore", "ROUGE"]
    main()