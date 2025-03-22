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
from concurrent.futures import ThreadPoolExecutor

# Ensure the root directory is in Python's path 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
print(sys.path)

# Custom utility imports
from utils.setup_logger import setup_logger
logger = setup_logger()
from utils.util import load_yaml, setup_llm_and_embeddings, read_all_file_suffix_X
from Agents.evaluator_agent import ConversationEvaluator

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
    
    def run_batch(self, user_utterances: list, batch_size=2) -> list:
        """Process multiple learner responses in parallel."""
        texts = [{
            "summary": self.tutorial_document[0],
            "current_step": self.current_step,
            "instruction": self.tutorial_document[self.current_step],
            "user_utterance": utterance
        } for utterance in user_utterances]

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = list(executor.map(self.chain.run, texts))
        
        return results

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
    
    def run_batch(self, instructions: list, batch_size=2) -> list:
        """Run multiple learner responses in parallel."""
        texts = [{"instruction": t} for t in instructions]

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = list(executor.map(self.chain.run, texts))

        return results   

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
def teacher_node_batch(state):
    messages = state["messages"]
    batch_size = args.batch_size
    batch_utterances = [msg.content for msg in messages[-batch_size:]]
    teacher_responses = teacher_agent.run_batch(batch_utterances)
    return {"messages": [AIMessage(content=resp) for resp in teacher_responses]}

def learner_node_batch(state):
    messages = state["messages"]
    batch_size = args.batch_size
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

    # **Use ThreadPoolExecutor for parallel agent interaction**
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_teacher = executor.submit(TeacherLLMAgent(config_teacher, tutorial_document).run, "")
        future_learner = executor.submit(LearnerLLMAgent(config_learner).run, tutorial_document[1])

        teacher_response = future_teacher.result()
        learner_response = future_learner.result()
    
    conversation = [f"Teacher: {teacher_response}", f"Learner: {learner_response}"]
    # conversation = generate_single_conversation(tutorial_document=tutorial_document, config_teacher=config_teacher, config_learner=config_learner)
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

# Batch Processing for Methods
def process_method_batch(methods_batch, summary, source_tutorial_path):
    """Process multiple methods in batch using threading."""
    results = []
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = [executor.submit(process_method, method, i + 1, summary, source_tutorial_path) 
                   for i, method in enumerate(methods_batch)]
        results = [f.result() for f in futures if f.result()]
    return results

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

    with ThreadPoolExecutor(max_workers=args.processes) as executor:
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
                batch_size = args.batch_size

                # ** FIX: Reset `futures` per document **
                futures = []
                method_batches = [methods[i:i + batch_size] for i in range(0, len(methods), batch_size)]
                for method_batch in method_batches:
                    futures.append(executor.submit(process_method_batch, method_batch, summary, source_tutorial_path))   

                # ** Collect results only for this document **
                results = [item for f in futures for item in f.result() if item]
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
        'num_conversation': count_conversation,
        'total_evaluation': avg_scores,
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
    logger.info(f"Number of available CPUs per node={mp.cpu_count()}")
    # Get the actual number of CPUs allocated by Slurm
    logger.info(f"Number of CPUs allocated by Slurm={int(os.environ.get('SLURM_CPUS_PER_TASK', 1))}")
    
    # Argument parsing to various settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_doc", type=int, default=None, help="Debug by running on max_doc number of tutorials.")
    parser.add_argument("--config_teacher", type=str, default="conf/ollma-llama3.yaml", help="Config file for teacher agent.")
    parser.add_argument("--config_learner", type=str, default="conf/ollma-llama3.yaml", help="Config file for learner agent.")
    parser.add_argument("--config_evaluator", type=str, default="conf/ollma-llama3.yaml", help="Config file for evaluator agent")
    parser.add_argument("--processes", type=int, default=min(mp.cpu_count() or 8, 8), help="Number of mutliprocessing.")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of batch size.")    
    parser.add_argument("--skip_existing_gen", action="store_true", help="Skip the generation of exisiting conversation nor not.")
    args = parser.parse_args()

    config_teacher = load_yaml(args.config_teacher)
    config_learner = load_yaml(args.config_learner)
    config_evaluator = load_yaml(args.config_evaluator)

    # The config_file str 
    config_setting_str = f'T-{config_teacher['llm']['model']}_L-{config_learner['llm']['model']}_E-{config_evaluator['llm']['model']}'

    main()