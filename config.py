import argparse
from datetime import datetime
import multiprocessing as mp
from utils.util import load_yaml


EVAL_METRICS = ["Question Ratio", "Completion Achieved", "Diversity Score", 
                "Engagement", "Coherence", "Depth", "Relevance", "Progress", "Naturalness", "Truthfulness",
                "BLEU", "METEOR", "BERTScore", "ROUGE"]


# Argument parsing to various settings
parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=str, default=None, help="Get the job id from the slurm sbatch.")
parser.add_argument("--max_doc", type=int, default=None, help="Debug by running on max_doc number of tutorials.")
# parser.add_argument("--max_doc", type=int, default=2, help="Debug by running on max_doc number of tutorials.")
parser.add_argument("--max_interaction", type=int, default=100, help="The max steps of teacher-learner interactions.")
parser.add_argument("--config_teacher", type=str, default="conf/ollama-llama3.yaml", help="Config file for teacher agent.")
parser.add_argument("--config_learner", type=str, default="conf/ollama-llama3.yaml", help="Config file for learner agent.")
parser.add_argument("--config_evaluator", type=str, default="conf/ollama-llama3.yaml", help="Config file for evaluator agent")
parser.add_argument("--manager", type=str, default="agent", help="'agent' or 'workflow'")
parser.add_argument("--processes", type=int, default=min(mp.cpu_count() or 8, 8), help="Number of mutliprocessing.")
parser.add_argument("--batch_size", type=int, default=64, help="Number of batch size.")    
parser.add_argument("--skip_existing", action="store_true", help="Skip the generation of exisiting conversation nor not.")
parser.add_argument("--plot_agent_workflow", action="store_true", help="Plot the workflow of agents nor not.")
parser.add_argument("--start_doc_id", type=int, default=0, help="The id of start number of doc if the previous job stoped.") 
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