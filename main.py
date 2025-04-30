import os, sys
import json
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

# Ensure the project's root directory is in Python's path 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "."))) 
print(sys.path)
from config import EVAL_METRICS, OUTPUT_CONFIG, JOBID, config_teacher, args, config_files, logger
from utils.util import read_all_file_suffix_X, count_gpu_availability
from utils.db import *
if args.manager == 'workflow':
    from manager.multiple_agent_manager_workflow import process_batch_method
elif args.manager == 'agent':
    from manager.multiple_agent_manager_agentic import process_batch_method
else:
    print('Err: No such a type of manager.')

from dataset.education_dataset import create_dataloader

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


def process_batches(data_loader):
    """Process data batches and group results by document"""
    count_conversation = 0
    total_dialogs = []
    os.makedirs("result", exist_ok=True)
    # Load the previous dialogues if assigned
    if args.previous_dialogue_path and os.path.exists(args.previous_dialogue_path):
        with open(args.previous_dialogue_path, 'r') as fr:
            previous_dialogue_json = json.load(fr)
            assert [args.config_teacher, args.config_learner, args.config_evaluator] == previous_dialogue_json['config_files']
            count_conversation = len(previous_dialogue_json['total_conversations'])
            # Add exisiting dialogues 
            total_dialogs.extend(previous_dialogue_json['total_conversations'])
            logger.info(f'****** Loaded {count_conversation} previous conversations ******\n')

    for batch in tqdm(data_loader):
        batch_size = len(batch)
        try:
            # Process method batches using optimized batch inference
            batch_conversations, batch_evaluations = process_batch_method(batch)
        except Exception as e:
            print(f'ERR: batch inference err ({e}). Assign None to this batch.')
            batch_conversations, batch_evaluations = []*batch_size, []*batch_size
        # Add batch dialogues
        for input_, conv, eval_ in zip(batch, batch_conversations, batch_evaluations):
            total_dialogs.append(OrderedDict({
                **input_,
                "conversation": conv,
                "evaluation": eval_
            }))
            logger.info(f'===== (doc_id, method_id) ({input_['doc_id']}, {input_['method_id']}) =====\n')
            logger.info(f'----- Full Conversation {count_conversation} -----\n')
            logger.info('\n'.join(conv))
            logger.info(f'----- Evaluation Result {count_conversation} -----\n')
            logger.info(eval_)
        
        count_conversation += batch_size

    return total_dialogs, count_conversation


def save_consolidated_results(total_dialogs, count_conversation):
    """Save consolidated results into a single JSON file"""

    avg_scores = calculate_average_scores(total_dialogs) if total_dialogs else {}

    dialogue_json = {
        'config_files': (args.config_teacher, args.config_learner, args.config_evaluator),
        'num_conversation': count_conversation,
        'total_evaluation': avg_scores,
        'total_conversations': total_dialogs
    }
    
    os.makedirs("result", exist_ok=True)
    final_all_dialogue_path = f'result/{OUTPUT_CONFIG}_{JOBID}.json'      
    logger.info(f"Save all dialogue to the path: {final_all_dialogue_path}")  
    with open(final_all_dialogue_path, 'w') as f:
        json.dump(dialogue_json, f, indent=4)


# Main Execution
def main():
    mdir = config_teacher['params']['root_doc_dir']  # './data/wikihow'
    docs = read_all_file_suffix_X(mdir=mdir, suffix='.json')

    # Use DataLoader to efficiently load and process data
    data_loader = create_dataloader(docs, batch_size=args.batch_size)
    logger.info(f"****** Processed {len(data_loader)} new batches of data in total ******")

    # Process data batches and group results by document
    total_dialogs, count_conversation = process_batches(data_loader)
    # print(total_dialogs)
    
    # Save consolidated results
    save_consolidated_results(total_dialogs, count_conversation)

    logger.info(f"****** Processed Conversation Number # new {len(data_loader.dataset)} over all {count_conversation} ******")



if __name__ == "__main__":
    # Detect GPUs
    num_gpu = count_gpu_availability()
    main()