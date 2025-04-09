import os, sys
import glob
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from config import args

# Ensure the project's root directory is in Python's path 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "."))) 
print(sys.path)
from config import EVAL_METRICS, OUTPUT_CONFIG, JOBID, config_teacher, args, config_files, logger
from utils.util import read_all_file_suffix_X, count_gpu_availability
from utils.db import *
from manager.multiple_agent_manager import process_batch_method

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

def load_existing_dialogues(docs, skip_existing, start_doc_id):
    """Efficiently load existing dialogues from files to avoid recomputation"""
    total_dialogs = defaultdict(list)
    count_conversation = 0
    
    if skip_existing:
        for doc_id, doc in enumerate(docs):
            source_tutorial_path = doc['path']
            target_dialogue_path = source_tutorial_path.replace('.json', f'.{OUTPUT_CONFIG}.txt')
            
            if glob.glob(target_dialogue_path) and doc_id <= start_doc_id:
                logger.info(f'Skip generation: Loading dialogs from the existing file {target_dialogue_path}.')
                with open(target_dialogue_path, 'r') as f:
                    dialogs_per_doc = json.load(f)

                if dialogs_per_doc:  # Only count if there are valid dialogues
                    count_conversation += len(dialogs_per_doc)

                total_dialogs[doc_id] = dialogs_per_doc # Grouping results by document

    return total_dialogs, count_conversation


def process_batches(data_loader, total_dialogs):
    """Process data batches and group results by document"""
    count_conversation = 0
    
    for batch in tqdm(data_loader):
        batch_size = len(batch)

        # Process method batches using optimized batch inference
        batch_conversations, batch_evaluations = process_batch_method(batch)

        for i in range(batch_size):
            if batch_conversations[i]:  # Avoid empty results
                count_conversation += 1
                doc_id = int(batch[i]["doc_id"])
                dialogue = {**batch[i],  # Restore original batch data with 'conversation' and 'evaluation' as the results
                            "conversation": batch_conversations[i],
                            "evaluation": batch_evaluations[i]}
                total_dialogs[doc_id].append(dialogue)
                # Save each dialogue to db
                save_dialogue_to_config_db(dialogue_id=batch[i]['conversation_id'], dialogue_data=dialogue, config_files=config_files)
                logger.info(f'----- Full Conversation {count_conversation} -----\n')
                logger.info('\n'.join(batch_conversations[i]))
                logger.info(f'----- Evaluation Result {count_conversation} -----\n')
                logger.info(batch_evaluations[i])

    return total_dialogs, count_conversation


def save_consolidated_results(total_dialogs, count_conversation):
    """Save consolidated results into a single JSON file"""
    # Flattening the defaultdict into a plain list
    list_total_dialogs =  [item for sublist in total_dialogs.values() for item in sublist] 

    avg_scores = calculate_average_scores(list_total_dialogs) if list_total_dialogs else {}

    dialogue_json = {
        'config_files': (args.config_teacher, args.config_learner, args.config_evaluator),
        'num_conversation': count_conversation,
        'total_evaluation': avg_scores,
        'total_conversations': list_total_dialogs
    }
    
    final_all_dialogue_path = f'result/{OUTPUT_CONFIG}_{JOBID}.json'      
    logger.info(f"Save all dialogue to the path: {final_all_dialogue_path}")  
    with open(final_all_dialogue_path, 'w') as f:
        json.dump(dialogue_json, f, indent=4)


# Main Execution
def main():
    mdir = config_teacher['params']['root_doc_dir']  # './data/wikihow'
    docs = read_all_file_suffix_X(mdir=mdir, suffix='.json', max_doc=args.max_doc)

    # Load dialogs if required
    total_dialogs, count_conversation = load_existing_dialogues(docs, args.skip_existing, args.start_doc_id)

    # Use DataLoader to efficiently load and process data
    data_loader = create_dataloader(docs, batch_size=args.batch_size, output_config=OUTPUT_CONFIG, skip_existing=args.skip_existing, start_doc_id=args.start_doc_id)
    logger.info(f"****** Processed {len(data_loader)} batches of data in total ******")

    # Process data batches and group results by document
    total_dialogs, count_conversation = process_batches(data_loader, total_dialogs)
    # print(total_dialogs)
    
    # Save consolidated results
    save_consolidated_results(total_dialogs, count_conversation)

    logger.info(f"****** Processed Conversation Number # {len(data_loader.dataset)} over all {count_conversation} ******")



if __name__ == "__main__":
    # Detect GPUs
    num_gpu = count_gpu_availability()
    main()