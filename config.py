import argparse
from datetime import datetime
import multiprocessing as mp
from utils.util import load_yaml, setup_logger

EVAL_METRICS = ["Question Ratio", "Completion Achieved", "Diversity Score", 
                "Clarity", "Engagement", "Coherence", "Depth", "Relevance", "Progress", "Naturalness", "Truthfulness",
                "BLEU", "METEOR", "BERTScore", "ROUGE"]

metrics = ["Clarity", "Engagement", "Coherence", "Depth", "Relevance", "Progress", "Naturalness", "Truthfulness"]

# Argument parsing to various settings
parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=str, default=None, help="Get the job id from the slurm sbatch.")
parser.add_argument("--max_dialog", type=int, default=None, help="Debug by running on max_dialog number of dialogues.")
parser.add_argument("--max_interaction", type=int, default=40, help="The max steps of teacher-learner interactions.")
parser.add_argument("--config_teacher", type=str, default="conf/ollama-llama3.yaml", help="Config file for teacher agent.")
parser.add_argument("--config_learner", type=str, default="conf/ollama-llama3.yaml", help="Config file for learner agent.")
parser.add_argument("--config_evaluator", type=str, default="conf/ollama-llama3.yaml", help="Config file for evaluator agent")
parser.add_argument("--manager", type=str, default="workflow", help="'agent' or 'workflow'")
parser.add_argument("--processes", type=int, default=min(mp.cpu_count() or 8, 8), help="Number of mutliprocessing.")
parser.add_argument("--batch_size", type=int, default=16, help="Number of batch size.")    
parser.add_argument("--plot_agent_workflow", action="store_true", help="Plot the workflow of agents nor not.") 
parser.add_argument("--previous_dialogue_path", type=str, default=None, help="The previous json file with generated dialogues.") 
args = parser.parse_args()

config_teacher = load_yaml(args.config_teacher)
config_learner = load_yaml(args.config_learner)
config_evaluator = load_yaml(args.config_evaluator)

config_files = [args.config_teacher, args.config_learner, args.config_evaluator]

# The config_file str
# TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M%S")  # YYMMDD_HHMMSS format 
JOBID = args.job_id
OUTPUT_CONFIG = f'T-{config_teacher['llm']['model']}_L-{config_learner['llm']['model']}_E-{config_evaluator['llm']['model']}'

PINK_COLOR = '\033[38;2;255;105;180m'  # RGB (255, 105, 180) for pink
RESET_COLOR = '\033[0m'  # Reset color back to normal

# Procedural control tokens
BEGIN = 'BEGIN'
NEXT = 'NEXT'
FINISHED = 'FINISHED'

logger = setup_logger(log_file=f'{args.job_id}.out')