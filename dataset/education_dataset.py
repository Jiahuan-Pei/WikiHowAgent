import os
import json
import uuid
from torch.utils.data import Dataset, DataLoader
from config import args
# from pprint import pprint

# json_data = {
#     "title": title,
#     "path": json_path,
#     "categories": categories,
#     "introduction": intro,
#     "methods": methods,
#     "qa": qa_items,
#     "tips": tips,
#     "warnings": warnings,
#     "things_youll_need": things_needed,
#     "references": references
# }

class DatasetWrapper(Dataset):
    """Wraps multiple tutorial documents for batch inference."""
    def __init__(self, docs: list, max_dialog=args.max_dialog, previous_dialogue_path=args.previous_dialogue_path):
        """
        Args:
            docs (list): List of tutorial documents.
            max_dialog (int): Max number of new dialogues to be loaded.
            previous_dialogue_path (str): load the previous dialogue if the path is not None
        """
        self.data = []
        count_dialogue = 0

        for doc_id, doc in enumerate(docs):
            # ✅ Skip already processed files if required) 
            previous_doc_method_pairs = []
            if previous_dialogue_path and os.path.exists(previous_dialogue_path):
                with open(previous_dialogue_path, 'r') as fr:
                    previous_dialogue_json = json.load(fr)
                    assert [args.config_teacher, args.config_learner, args.config_evaluator] == previous_dialogue_json['config_files']
                    # pprint(previous_dialogue_json['total_conversations'][0], indent=4)
                    previous_doc_method_pairs = [(d['doc_id'], d['method_id']) for d in previous_dialogue_json['total_conversations']]

            # ✅ Each method has one data for one conversation
            for method_id, method in enumerate(doc["methods"]):
                if (doc_id, method_id) in previous_doc_method_pairs:
                    print(f'DEBUG: skip ({doc_id}, {method_id})')
                    continue # Skip this data
                elif (isinstance(method, dict) and method["steps"]) or (isinstance(method, list) and method):
                    self.data.append({
                        "conversation_id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "method_id": method_id,
                        "source_tutorial_path": str(doc['path']), # Unique ID
                        "title": doc["title"],
                        "categories": doc["categories"],
                        "summary": doc["introduction"],
                        "tutorial": method
                    })
                    count_dialogue+=1
                else:
                    pass
                # If current dialogue id is larger than the max_dialog, then break
                if max_dialog and count_dialogue >= max_dialog:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """Custom collate function that returns a list of dictionaries instead of stacking tensors."""
    return batch  # No transformation, just return as-is

# Use the custom collate_fn in your DataLoader
def create_dataloader(docs, batch_size, num_workers=4):
    """Creates a DataLoader with custom collate_fn."""
    dataset = DatasetWrapper(docs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn) # One doc per batch
