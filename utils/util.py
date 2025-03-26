import os, sys
import getpass
import yaml
import json
import torch
import logging
import os
import yaml
import torch
import multiprocessing as mp

from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Ensure the root directory is in Python's path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

# Ensure API Key is set
def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

_set_if_undefined("OPENAI_API_KEY")

def count_gpu_availability():
    # Check if CUDA is available
    print("CUDA is available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        # Get the number of GPUs
        print("Number of GPUs:", torch.cuda.device_count())
        
        # Get current GPU device
        print("Current GPU device:", torch.cuda.current_device())
        
        # Get GPU name
        print("GPU device name:", torch.cuda.get_device_name(0))
        
        # Test GPU computation
        x = torch.rand(5, 3)
        print("Input tensor:", x)
        
        # Move tensor to GPU
        x = x.cuda()
        print("Tensor on GPU:", x)
        print("Tensor device:", x.device)
        return torch.cuda.device_count()
    else:
        return 0    

def count_cpu_availability():
    """Return the number of available CPUs."""
    # Get the total CPUs on the node (for information only)
    logger.info(f"Number of available CPUs per node={mp.cpu_count()}")
    # Get the actual number of CPUs allocated by Slurm
    logger.info(f"Number of CPUs allocated by Slurm={int(os.environ.get('SLURM_CPUS_PER_TASK', 1))}")
    return 

def load_yaml(config_file='conf/ollama-llama3.yaml'):
    """Load config from a YAML configuration file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_llm_and_embeddings(config):

    llm_config = config['llm']
    # embedding_config = config['embeddings']
    
    # Detect GPUs
    num_gpu = count_gpu_availability()
    logger.info(f"Detected {num_gpu} available GPUs")
    
    if num_gpu > 0:
        os.environ["OLLAMA_ACCELERATE"] = "1"
        os.environ["OLLAMA_NUM_GPU"] = str(num_gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpu))
        logger.info(f"Running model with {num_gpu} GPUs.")
    else:
        os.environ["OLLAMA_ACCELERATE"] = "0"
        logger.warning("No GPUs detected, running on CPU.")

    # Initialize LLM and embeddings
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
            verbose=True,
            timeout=600,
            num_ctx=8192,
            disable_streaming=False, # ✅ Ensure streaming is enabled; With streaming: Tokens are processed in parallel, increasing efficiency.
            # num_gpu=num_gpu,  # ✅ Ensure GPU acceleration
            num_thread=16  # ✅ More threads for efficiency
        )
        embeddings = OllamaEmbeddings(model=llm_config['model'])
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

def get_snellious_job_id():
    """
    Function to retrieve the job ID from Snellious.
    This should be replaced with the actual Snellious API call or environment variable.
    """
    job_id = os.getenv("SNELLIOUS_JOB_ID")  # Example: Fetch job ID from environment
    if not job_id:
        job_id = 'default'  # Get the current script name
    return job_id


def setup_logger(log_file=None, log_level=logging.INFO):
    """
    Set up a logger that logs messages to both the console and a file.

    Parameters:
        log_file (str): The name of the file where logs will be saved.
        log_level (int): The logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
        logger (logging.Logger): Configured logger instance.
    """

    # Create a custom logger
    logger = logging.getLogger("AppLogger")
    logger.setLevel(log_level)

    # Prevent duplicate logs if the logger is already set up
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define the log format
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    if log_file is None:
        log_file = f"{get_snellious_job_id()}.out"
    else:
        log_file = f"{Path(log_file).stem}.log"
        
    # File handler to log to a file
    os.makedirs('log', exist_ok=True)
    log_file = os.path.join('log', log_file)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(log_format)

    # Console handler to log to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logger initialized. Logging to file: %s", log_file)
    return logger

logger = setup_logger()