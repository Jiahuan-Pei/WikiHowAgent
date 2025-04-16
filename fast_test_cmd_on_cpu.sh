# Step 1: create a config file under folder conf/
# Step 2: run your python script with small data
python main.py --max_doc=2 --job_id='011' --config_teacher conf/ollama-mistral.yaml --config_learner conf/ollama-mistral.yaml --config_evaluator conf/ollama-mistral.yaml
# Step 3: check your ouptut file under folder result/