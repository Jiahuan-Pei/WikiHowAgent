import sqlite3
import json
import hashlib
import os

DATABASE_PATH = "data/dialogues.db"

# Function to generate a valid table name based on the combination of config files
def get_table_name_from_config_files(config_files):
    """
    Create a unique table name based on the hash of the sorted list of config files.
    This ensures the combination of config files generates a unique table.
    """
    # Sort the config files to ensure that the order doesn't affect the hash
    sorted_files = sorted(config_files)
    # Generate a hash from the sorted config files
    config_combination_hash = hashlib.md5(' '.join(sorted_files).encode('utf-8')).hexdigest()
    return f"dialogues_{config_combination_hash}"

# Function to create a table for a specific config file combination (if not already exists)
def create_dialogue_table_for_config_combination(config_files):
    table_name = get_table_name_from_config_files(config_files)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            dialogue_id TEXT PRIMARY KEY,
            dialogue_json TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Function to save a dialogue to a config file combination-specific table
def save_dialogue_to_config_db(dialogue_id, dialogue_data, config_files):
    """
    Save a dialogue to the table specific to the combination of config files.
    """
    table_name = get_table_name_from_config_files(config_files)
    dialogue_json = json.dumps(dialogue_data)

    # Create the table if it doesn't exist
    create_dialogue_table_for_config_combination(config_files)

    # Insert or replace the dialogue into the specific config table
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(f'''
        INSERT OR REPLACE INTO {table_name} (dialogue_id, dialogue_json)
        VALUES (?, ?)
    ''', (dialogue_id, dialogue_json))
    conn.commit()
    print(f"Saved dialogue {dialogue_id} to the table {table_name}" + '.'*20)
    conn.close()

# Function to query dialogues from a config file combination-specific table
def query_dialogues_from_config(criteria, config_files):
    """
    Query dialogues from the table associated with the config file combination.
    """
    table_name = get_table_name_from_config_files(config_files)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(f'SELECT dialogue_id, dialogue_json FROM {table_name}')
    rows = cursor.fetchall()

    matching_dialogues = []
    for row in rows:
        dialogue_id, dialogue_json = row
        dialogue = json.loads(dialogue_json)  # Convert JSON string back to a dictionary
        
        # Apply filtering criteria
        match = True
        for key, value in criteria.items():
            if key in dialogue:
                if dialogue[key] != value:
                    match = False
                    break
            else:
                match = False
                break
        
        if match:
            matching_dialogues.append(dialogue)
    
    conn.close()
    return matching_dialogues

# Example: Restore a dialogue using its ID from a specific config file combination
def restore_dialogue_from_config(dialogue_id, config_files):
    table_name = get_table_name_from_config_files(config_files)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(f'SELECT dialogue_json FROM {table_name} WHERE dialogue_id = ?', (dialogue_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])  # Parse the JSON string back to a Python object
    else:
        return None

# Example usage of the functions
def demo():

    # List of config files representing a combination
    config_files = [
        "conf/ollama-llama3.yaml",
        "conf/ollama-llama3.yaml",
        "conf/ollama-llama3.yaml"
    ]

    # Sample dialogue data
    sample_dialogue = {
        "doc_id": "0",
        "method_id": "1",
        "steps": ["Step 1: Introduction", "Step 2: Tutorial", "Step 3: Conclusion"],
        "categories": ["Home and Garden", "Housekeeping"]
    }

    # Save dialogues for the combination of config files
    save_dialogue_to_config_db("dialogue_001", sample_dialogue, config_files)

    # Query the dialogues for a specific config file combination
    criteria = {
        "doc_id": "0",
        "method_id": "1"
    }
    matching_dialogues = query_dialogues_from_config(criteria, config_files)

    # Print the matching dialogues
    for dialogue in matching_dialogues:
        print(dialogue)

    # Restore a dialogue from a specific config file combination
    restored_dialogue = restore_dialogue_from_config("dialogue_001", config_files)
    print("Restored Dialogue:", restored_dialogue)

# demo()