import os
import uuid
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, docs: list, output_config: str, skip_existing: bool=False, start_doc_id: int=0):
        """
        Args:
            docs (list): List of tutorial documents.
            output_config (str): Output file format.
            skip_existing (bool): Whether to skip already processed dialogues.
            start_doc_id (int): ID to start processing from.
        """
        self.data = []

        for doc_id, doc in enumerate(docs):
            source_tutorial_path = doc['path']
            target_dialogue_path = source_tutorial_path.replace('.json', f'.{output_config}.txt')
            
            # ✅ Skip already processed files if required (skip_existing=True and current doc_id is smaller) 
            if os.path.exists(target_dialogue_path) and skip_existing and doc_id <= start_doc_id: 
                continue  # Skip this document

            # ✅ Each method has one data for one conversation
            for method_id, method in enumerate(doc["methods"]):
                if (isinstance(method, dict) and method["steps"]) or (isinstance(method, list) and method):
                    self.data.append({
                        "conversation_id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "method_id": method_id,
                        "source_tutorial_path": str(source_tutorial_path), # Unique ID
                        "title": doc["title"],
                        "categories": doc["categories"],
                        "summary": doc["introduction"],
                        "tutorial": method
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """Custom collate function that returns a list of dictionaries instead of stacking tensors."""
    return batch  # No transformation, just return as-is

# Use the custom collate_fn in your DataLoader
def create_dataloader(docs, batch_size, output_config, skip_existing, start_doc_id, num_workers=4):
    """Creates a DataLoader with custom collate_fn."""
    dataset = DatasetWrapper(docs, output_config, skip_existing, start_doc_id)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn) # One doc per batch
