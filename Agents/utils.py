import os
import yaml
import json
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def setup_llm_and_embeddings(config_file='conf/ollma-llama3.yaml'):
    """Load LLM and embeddings from a YAML configuration file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    llm_config = config['llm']
    embedding_config = config['embeddings']
    param_config  = config['params']

    # Initialize LLM and embeddings based on the configuration
    if llm_config['model_type'] == "openai":
        llm = ChatOpenAI(
            model=llm_config['model'], 
            temperature=llm_config['temperature']
        )
        embeddings = OpenAIEmbeddings()
    elif llm_config['model_type'] == "ollama":
        llm = ChatOllama(
            model=llm_config['model'],
            temperature=llm_config['temperature'],
            verbose=True,  # You can also make this configurable
            timeout=600,
            num_ctx=8192,
            disable_streaming=False
        )
        embeddings = OllamaEmbeddings(model=embedding_config['model'])
    else:
        raise ValueError(f"Unsupported model type: {llm_config['model_type']}")

    return llm, embeddings, config 

def read_all_file_suffix_X(mdir='./data/wikihow', suffix='.json', max_doc=None):    
    docs = []
    count = 0
    for topic in os.listdir(mdir):
        topic_path = os.path.join(mdir, topic)
        if not os.path.isdir(topic_path):
            continue
        for task in os.listdir(topic_path):
            task_path = os.path.join(topic_path, task)
            if not os.path.isdir(task_path):
                continue
            for file in os.listdir(task_path):             
                if file.endswith(suffix):  # Filter files by the specified suffix
                    file_path = os.path.join(task_path, file)
                    # print(file_path)  # Print the file path
                    # Add logic to read the file and append to docs
                    count+=1
                    if max_doc is not None and count >= max_doc:
                        return docs
                    with open(file_path, 'r') as f:
                        if suffix == '.json':                   
                                json_data = json.load(f)
                                docs.append(json_data)
                        else:
                                docs.append(f.read())
    return docs