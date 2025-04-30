#!/bin/bash
#SBATCH --job-name=conversation_generation
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err
#SBATCH --time=5-00:00:00        # Max runtime
#SBATCH --nodes=1                # Use #number physical machines
#SBATCH --ntasks=1               # 🔥 Run #number parallel python scripts when you have different settings
#SBATCH --gres=gpu:1             # Request #number GPU, when you need more control over GPU type or specific features  (A100)
#SBATCH --cpus-per-task=8        # 🔥 Assign #number CPUs per task; Match with args.processes=8; If inference is GPU-bound, having too many CPU processes won't help.

#SBATCH --mem=16GB               # Request of memory
#SBATCH --partition=gpu_h100     # Use the GPU partition

echo "Starting job on $(hostname) at $(date)"
echo "Total CPUs allocated: $SLURM_JOB_CPUS_PER_NODE"
echo "Number of CPUs allocated by Slurm=$SLURM_CPUS_PER_TASK"

# --- 1. Load environment ---
module load 2024
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0
module load Python/3.12.3-GCCcore-13.3.0

source ~/.bashrc
source activate worldtaskeval

# --- 2. Set Environment Variables (for GPU) ---
export PATH="~/anaconda3/envs/worldtaskeval/bin:$PATH" # Use the virtual env worldtaskeval
export NLTK_DATA="/gpfs/home3/jpei1/nltk_data"
# GPU Optimization
# export CUDA_VISIBLE_DEVICES=0  # Use the allocated GPU
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

export OLLAMA_FORCE_CUDA=1
export OLLAMA_NUM_GPU=1        # Match the number of requested GPUs
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export OLLAMA_NUM_GPU=4
export OLLAMA_ACCELERATE=1
export OLLAMA_LLM_LIBRARY=cublas  # Use cuBLAS for acceleration
# export OLLAMA_LLM_LIBRARY=flash-attn
export OLLAMA_PRECISION=fp16  # Enable 16-bit floats for Tensor Cores # A100 GPUs perform best with mixed precision:
export OLLAMA_BATCH_SIZE=32  # Adjust based on GPU memory
export OLLAMA_NUM_THREAD=16
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OLLAMA_USE_CUDA_GRAPHS=1
export OLLAMA_CONTEXT_SIZE=8192  # Increase from default (~2048)
export OLLAMA_KEEP_LOADED=1
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
alias python=~/anaconda3/envs/worldtaskeval/bin/python

# --- 2. Start Ollama Servers for Two Models ---
echo "Starting Ollama server..."
MODEL_NAME_1="openchat"
MODEL_NAME_2="gemma"
# Dynamically assign a port based on the job ID
PORT1=$((11434 + ($SLURM_JOB_ID % 1000)))
PORT2=$((PORT1 + 1))
# Directory for model cache
OLLAMA_DIR_1="$HOME/.ollama_${SLURM_JOB_ID}_1"
OLLAMA_DIR_2="$HOME/.ollama_${SLURM_JOB_ID}_2"

# Create isolated model/data directory
mkdir -p "$OLLAMA_DIR_1"
mkdir -p "$OLLAMA_DIR_2"

# Start first Ollama server
export OLLAMA_HOST=0.0.0.0:$PORT1
export SINGULARITYENV_OLLAMA_HOST=$OLLAMA_HOST
# Optional: Bind a persistent ollama model cache dir
export OLLAMA_DIR="$OLLAMA_DIR_1"

singularity exec --nv \
  --bind "$OLLAMA_DIR_1":/root/.ollama \
  --bind "$NLTK_DATA:/root/nltk_data" \
  ollama_latest.sif ollama serve&

OLLAMA_PID1=$!

# Start second Ollama server
sleep 5  # slight delay to avoid collision
export OLLAMA_HOST=0.0.0.0:$PORT2
export SINGULARITYENV_OLLAMA_HOST=$OLLAMA_HOST

singularity exec --nv \
  --bind "$OLLAMA_DIR_2":/root/.ollama \
  --bind "$NLTK_DATA:/root/nltk_data" \
  ollama_latest.sif ollama serve &

OLLAMA_PID2=$!

# Wait until both servers are ready
until curl -s http://localhost:$PORT1/api/tags > /dev/null; do
    sleep 5
done
echo "First Ollama server ready at port $PORT1"

until curl -s http://localhost:$PORT2/api/tags > /dev/null; do
    sleep 5
done
echo "Second Ollama server ready at port $PORT2"

# --- 3. Pull both models ---
export OLLAMA_HOST=0.0.0.0:$PORT1
singularity exec --nv \
  --bind "$OLLAMA_DIR_1":/root/.ollama \
  --bind "$NLTK_DATA:/root/nltk_data" \
  ollama_latest.sif ollama pull $MODEL_NAME_1
echo "Model $MODEL_NAME_1 pulled."

export OLLAMA_HOST=0.0.0.0:$PORT2
singularity exec --nv \
  --bind "$OLLAMA_DIR_2":/root/.ollama \
  --bind "$NLTK_DATA:/root/nltk_data" \
  ollama_latest.sif ollama pull $MODEL_NAME_2
echo "Model $MODEL_NAME_2 pulled."

# --- 4. Run the Python Script with Correct Python Path ---
export PYTHONPATH=$PWD  # Ensure Python finds your package
~/anaconda3/envs/worldtaskeval/bin/python main.py --processes $SLURM_CPUS_PER_TASK --job_id $SLURM_JOB_ID --batch_size 16 --config_teacher conf/ollama-openchat.yaml --config_learner conf/ollama-gemma.yaml --config_evaluator conf/ollama-openchat.yaml

# --- 5. Cleanup: Kill Ollama Server ---
echo "Cleaning up servers..."
kill $OLLAMA_PID1
kill $OLLAMA_PID2

echo "Job completed at $(date)"
