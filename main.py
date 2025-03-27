import os, sys
import glob
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Ensure the root directory is in Python's path 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
print(sys.path)
from config import EVAL_METRICS, OUTPUT_CONFIG, JOBID, config_teacher, args
from utils.util import read_all_file_suffix_X, logger
from manager.mutiple_agent_workflow import process_method_batch

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
# def main():
#     mdir = config_teacher['params']['root_doc_dir'] # './data/wikihow'

#     count_conversation = 0
#     docs = read_all_file_suffix_X(mdir=mdir, suffix='.json', max_doc=args.max_doc)

#     total_dialogs = []

#     doc_id = 0
#     # Runs multiple documents in parallel.
#     with ThreadPoolExecutor(max_workers=args.processes) as executor:
#         for doc in tqdm(docs):
#             source_tutorial_path = doc['path']
#             target_dialogue_path = source_tutorial_path.replace('.json', f'.{OUTPUT_CONFIG}.txt')

#             # Dialogs per doc
#             dialogs_per_doc = []
#             if glob.glob(target_dialogue_path) and args.skip_existing_gen and doc_id <= args.start_doc_id:
#                 logger.info(f'Skip generation: Loading dialogs from the exisiting file {target_dialogue_path}.')
#                 with open(target_dialogue_path, 'r') as f:
#                     dialogs_per_doc = json.load(f)
#                 total_dialogs.extend(dialogs_per_doc) 
#                 # Update conversation count and metrics
#                 count_conversation += len(dialogs_per_doc)         
#             else:
#                 summary = doc['introduction']
#                 methods = doc['methods']
#                 batch_size = args.batch_size

#                 # ** FIX: Reset `futures` per document **
#                 futures = []
#                 method_batches = [methods[i:i + batch_size] for i in range(0, len(methods), batch_size)]
#                 for method_batch in method_batches:
#                     futures.append(executor.submit(process_method_batch, method_batch, summary, source_tutorial_path))   

#                 # ** Collect results only for this document **
#                 dialogs_per_doc = [item for f in futures for item in f.result() if item]
#                 # Update counts and collect results
#                 count_conversation += len(dialogs_per_doc)
#                 logger.info(f"****** Processed Conversation Number # {count_conversation} ******")
                
#                 # Save generated dialogues per doc
#                 with open(target_dialogue_path, 'w') as f:
#                     json.dump(dialogs_per_doc, f, indent=4)               
#                 # dialogs_per_doc.extend(results)
#                 total_dialogs.extend(dialogs_per_doc)

#             doc_id += 1

#     # Compute average scores
#     avg_scores = calculate_average_scores(total_dialogs)

#     # Save all dialogs together in one file
#     dialouge_json = {
#         'config_files': (args.config_teacher, args.config_learner, args.config_evaluator),
#         'num_conversation': count_conversation,
#         'total_evaluation': avg_scores,
#         'total_conversations': total_dialogs
#     }
#     final_all_dialogue_path = f'result/{OUTPUT_CONFIG}_{JOBID}.json'      
#     logger.info(f"Save all dialogue to the path: {final_all_dialogue_path}")  
#     with open(final_all_dialogue_path, 'w') as f:
#         json.dump(dialouge_json, f, indent=4) 

#     logger.info(f'Generated {count_conversation} conversations!')

# Main Execution
def main():
    mdir = config_teacher['params']['root_doc_dir']  # './data/wikihow'

    count_conversation = 0
    docs = read_all_file_suffix_X(mdir=mdir, suffix='.json', max_doc=args.max_doc)

    total_dialogs = []

    doc_id = 0
    # Runs multiple documents in parallel.
    with ThreadPoolExecutor(max_workers=args.processes) as executor:
        for doc in tqdm(docs):
            source_tutorial_path = doc['path']
            target_dialogue_path = source_tutorial_path.replace('.json', f'.{OUTPUT_CONFIG}.txt')

            # Dialogs per doc
            dialogs_per_doc = []
            if glob.glob(target_dialogue_path) and args.skip_existing_gen and doc_id <= args.start_doc_id:
                logger.info(f'Skip generation: Loading dialogs from the existing file {target_dialogue_path}.')
                with open(target_dialogue_path, 'r') as f:
                    dialogs_per_doc = json.load(f)

                if dialogs_per_doc:  # ✅ Only count if there are valid dialogues
                    with lock:
                        count_conversation += len(dialogs_per_doc)

                total_dialogs.extend(dialogs_per_doc)

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
                dialogs_per_doc = []
                for f in futures:
                    try:
                        res = f.result()  # ✅ Handle potential errors
                        if res:
                            dialogs_per_doc.extend(res)
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")

                if dialogs_per_doc:  # ✅ Avoid adding empty results
                    with lock:
                        count_conversation += len(dialogs_per_doc)

                logger.info(f"****** Processed Conversation Number # {count_conversation} ******")

                # Save generated dialogues per doc
                with open(target_dialogue_path, 'w') as f:
                    json.dump(dialogs_per_doc, f, indent=4)

                total_dialogs.extend(dialogs_per_doc)

            doc_id += 1

    # Compute average scores
    avg_scores = calculate_average_scores(total_dialogs) if total_dialogs else {}

    # Save all dialogs together in one file
    dialogue_json = {
        'config_files': (args.config_teacher, args.config_learner, args.config_evaluator),
        'num_conversation': count_conversation,
        'total_evaluation': avg_scores,
        'total_conversations': total_dialogs
    }
    final_all_dialogue_path = f'result/{OUTPUT_CONFIG}_{JOBID}.json'      
    logger.info(f"Save all dialogue to the path: {final_all_dialogue_path}")  
    with open(final_all_dialogue_path, 'w') as f:
        json.dump(dialogue_json, f, indent=4)

    logger.info(f'Generated {count_conversation} conversations!')


if __name__ == "__main__":
    # Initialize a lock for thread safety
    lock = Lock()
    main()