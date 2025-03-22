import os, sys
import json
import uuid
import re
import glob
import numpy as np
import argparse
from tqdm import tqdm
from typing import Annotated
from typing_extensions import TypedDict
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
# from langchain.schema.runnable import RunnableSequence

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

# Custom utility imports
from setup_logger import setup_logger
logger = setup_logger()
from util import load_yaml, setup_llm_and_embeddings, read_all_file_suffix_X
from evaluator_agent import ConversationEvaluator

import multiprocessing as mp

# Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Teacher Agent
class TeacherLLMAgent:
    def __init__(self, config, tutorial_document=None):
        self.tutorial_document = tutorial_document
        self.current_step = 1
        self.total_steps = len(tutorial_document) - 1  # 0th index is the summary
        self.llm, _, self.config = setup_llm_and_embeddings(config)

        self.prompt_template = PromptTemplate(
            input_variables=["current_step", "instruction", "user_utterance"],
            template=(
                "You are a teacher guiding a learner through a tutorial. "
                "Your job is to give instructions for the current step and answer any questions."
                "Summary: {summary}\n"
                "Instruction (Step {current_step}): {instruction}\n"
                "Learner: {user_utterance}\n"
                "Teacher:"
            ),
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=True)
        # self.chain = self.prompt_template | self.llm
        # self.chain =  RunnableSequence(self.prompt_template, self.llm)

    def run(self, user_utterance: str) -> str:
        """Process the learner's response and generate the teacher's reply."""
        if self.current_step > self.total_steps:
            return "FINISHED"

        summary = self.tutorial_document[0]  # Summary of the tutorial
        instruction = self.tutorial_document[self.current_step]

        # Generate teacher response
        # teacher_response = self.chain.invoke({
        #     "summary": summary,
        #     "current_step": self.current_step,
        #     "instruction": instruction,
        #     "user_utterance": user_utterance
        # })
        teacher_response = self.chain.run(summary=summary, 
                                          current_step=self.current_step,
                                          instruction=instruction,
                                          user_utterance=user_utterance)       

        # Check if the learner asked a question
        is_question = "?" in user_utterance
        if is_question:
            return teacher_response  # Answer question without moving forward

        # Determine if the learner wants to proceed
        next_step_decision = self.llm.invoke(
            f"Based on this response: '{user_utterance}', does the learner understand and want to proceed? "
            f"Reply with 'yes' or 'no'."
        ).content.lower()

        if "yes" in next_step_decision:
            self.current_step += 1  # Move to the next step

        if self.current_step > self.total_steps:
            return f"Step FINAL: {teacher_response}\nThe tutorial is now complete. FINISHED."

        return teacher_response
    

# Learner Agent
class LearnerLLMAgent:
    def __init__(self, config):
        self.llm, _, self.config = setup_llm_and_embeddings(config)

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
        # self.chain = self.prompt_template | self.llm
        # self.chain = RunnableSequence(self.prompt_template, self.llm)

    def run(self, instruction: str) -> str:
        # return self.chain.invoke({"instruction": instruction})
        return self.chain.run(instruction=instruction)

# Conversation Nodes
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
def process_method(method, method_id, summary, source_tutorial_path, methods=None, config_teacher=None, config_learner=None, config_evaluator=None):
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


def calculate_average_scores(dialogs):
    """Calculate average scores for a list of dialog objects"""
    if not dialogs:
        return {}
    
    # Get all scores for each metric
    scores = {metric: [] for metric in EVAL_METRICS}
    for dialog in dialogs:
        for metric in EVAL_METRICS:
            if metric in dialog.get('evaluation', {}):
                scores[metric].append(dialog['evaluation'][metric])
    
    # Calculate averages
    return {metric: (np.mean(values), np.std(values)) if values else 0.0 for metric, values in scores.items()}

# Main Execution
def main():
    mdir = config_teacher['params']['root_doc_dir'] # './data/wikihow'

    count_conversation = 0
    docs = read_all_file_suffix_X(mdir=mdir, suffix='.json', max_doc=args.max_doc)

    total_dialogs = []

    for doc in tqdm(docs):
        source_tutorial_path = doc['path']
        target_dialogue_path = source_tutorial_path.replace('.json', '.txt')

        # Dialogs per doc
        dialogs = []
        if glob.glob(target_dialogue_path) and args.skip_existing_gen:
            logger.info(f'Skip generation: Loading dialogs from the exisiting file {target_dialogue_path}.')
            with open(target_dialogue_path, 'r') as f:
                dialogs = json.load(f)
            dialogs.extend(dialogs)
            total_dialogs.extend(dialogs) 
            # Update conversation count and metrics
            count_conversation += len(dialogs)         
        else:
            summary = doc['introduction']
            methods = doc['methods']
            # Single processing
            # for i, method in enumerate(methods):
            #     count_conversation += 1
            #     logger.info(f"\n--- Generating the Conversation Number: {count_conversation} ---\n")
            #     # Generate each dialog
            #     dialog = process_method(method, i + 1, summary, source_tutorial_path, None, logger=logger)
            #     dialogs.append(dialog)
            #     total_dialogs.append(dialog)

            # Multiprocessing
            # Create a list of arguments for each method
            method_args = [(method, i + 1, summary, source_tutorial_path, None, config_teacher, config_learner, config_evaluator) for i, method in enumerate(methods)]
            
            # Use multiprocessing to generate conversations in parallel
            with mp.Pool(processes=args.processes) as pool:
                results = pool.starmap(process_method, method_args)
            
            # Update counts and collect results
            count_conversation += len(results)
            logger.info(f"****** Processed Conversation Number # {count_conversation} ******")
            # Save generated dialogues per doc
            with open(target_dialogue_path, 'w') as f:
                json.dump(results, f, indent=4)               
            dialogs.extend(results)
            total_dialogs.extend(results)
    # Compute average scores
    avg_scores = calculate_average_scores(dialogs)

    # Save all dialogs together in one file
    dialouge_json = {
        'config_files': (args.config_teacher, args.config_learner, args.config_evaluator),
        'total_evaluation': avg_scores,
        'num_conversation': count_conversation,
        'total_conversations': total_dialogs
    }        
    with open(f'result/{config_setting_str}.json', 'w') as f:
        json.dump(dialouge_json, f, indent=4) 

    logger.info(f'Generated {count_conversation} conversations!')

if __name__ == "__main__":
    EVAL_METRICS = ["Question Ratio", "Completion Achieved", "Diversity Score", 
                "Engagement", "Coherence", "Depth", "Relevance", "Progress", "Naturalness", "Truthfulness",
                "BLEU", "METEOR", "BERTScore", "ROUGE"]

    # Get the total CPUs on the node (for information only)
    logger.info(f"Nnumber of available CPUs per node={mp.cpu_count()}")
    # Get the actual number of CPUs allocated by Slurm
    logger.info(f"Number of CPUs allocated by Slurm={int(os.environ.get('SLURM_CPUS_PER_TASK', 1))}")
    
    # Argument parsing to various settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_doc", type=int, default=None, help="Debug by running on max_doc number of tutorials.")
    parser.add_argument("--config_teacher", type=str, default="conf/ollma-llama3.yaml", help="Config file for teacher agent.")
    parser.add_argument("--config_learner", type=str, default="conf/ollma-llama3.yaml", help="Config file for learner agent.")
    parser.add_argument("--config_evaluator", type=str, default="conf/ollma-llama3.yaml", help="Config file for evaluator agent")
    parser.add_argument("--processes", type=int, default=min(mp.cpu_count() or 8, 8), help="Number of mutliprocessing.")
    parser.add_argument("--skip_existing_gen", action="store_true", help="Skip the generation of exisiting conversation nor not.")
    args = parser.parse_args()

    config_teacher = load_yaml(args.config_teacher)
    config_learner = load_yaml(args.config_learner)
    config_evaluator = load_yaml(args.config_evaluator)

    # The config_file str 
    config_setting_str = f'T-{config_teacher['llm']['model']}_L-{config_learner['llm']['model']}_E-{config_evaluator['llm']['model']}'

    main()