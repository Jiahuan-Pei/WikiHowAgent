# --- 1. Load required modules based on the specific experimental enviroment ---

# --- 2. Set Environment Variables (for GPU) ---
export PATH="~/anaconda3/envs/worldtaskeval/bin:$PATH" # Use the virtual env worldtaskeval
export NLTK_DATA="/gpfs/home3/jpei1/nltk_data"
# GPU Optimization
export CUDA_VISIBLE_DEVICES=0  # Use the allocated GPU
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

# --- 3. Activate Virtual Environment ---
source ~/.bashrc
source activate worldtaskeval

# --- 4. Singularity and GPU Access ---
# 4.1 Verify
singularity --version
# Check If Singularity Detects GPUs
singularity exec --nv ollama_latest.sif nvidia-smi
echo "Checking available executables inside Singularity:"
singularity exec --nv ollama_latest.sif ~/anaconda3/envs/worldtaskeval/bin/python -c "import torch; print('*'*20, torch.cuda.is_available(), torch.cuda.device_count())"
singularity exec --nv ollama_latest.sif echo $LD_LIBRARY_PATH
singularity exec --nv ollama_latest.sif which python
singularity exec --nv ollama_latest.sif which ollama
# Check If Ollama Supports Multi-GPU
# singularity exec --nv ollama_latest.sif ollama show

# 4.2 Start and test Ollama with GPU Support ---
echo "Starting Ollama server..."
singularity exec --nv ollama_latest.sif ollama serve &
# singularity exec --nv --bind ~/anaconda3/envs/worldtaskeval ollama_latest.sif ollama serve &
# singularity exec --nv --bind ~/anaconda3/envs/worldtaskeval ollama_latest.sif ollama serve &
OLLAMA_PID=$!
until curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 5
done
echo "Ollama server is ready!"
# 4.4 Pull LLM Model (Ensure Model is Downloaded) ---
singularity exec --nv ollama_latest.sif ollama pull llama3
echo "Ollama model is downloaded!"
singularity exec --nv ollama_latest.sif ollama run llama3 --verbose
echo "Test Ollama model inference time!"
time singularity exec --nv ollama_latest.sif ollama run llama3 "Explain quantum mechanics in 100 words."

# --- 5. Run the Python Script with Correct Python Path ---
export PYTHONPATH=$PWD  # Ensure Python finds your package
~/anaconda3/envs/worldtaskeval/bin/python main.py --processes $SLURM_CPUS_PER_TASK --job_id $SLURM_JOB_ID --batch_size=128  --config_teacher conf/openai-gpt.yaml --config_learner conf/openai-gpt.yaml --config_evaluator conf/openai-gpt.yaml