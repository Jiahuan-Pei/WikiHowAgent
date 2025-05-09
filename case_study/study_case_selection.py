import json
import numpy as np
import pandas as pd

metrics = ["Clarity", "Engagement", "Coherence", "Depth",
           "Relevance", "Progress", "Naturalness", "Truthfulness"]

def select_cases(json_files, k_positive, k_negative, fname):
    # Compute best and worst cases
    r = []
    for i in range(len(json_files)):
        with open(f"result/{json_files[i]}", "r") as f:
            data = json.load(f)
            mean_score_per_dialog = [np.mean([d["evaluation"][m] for m in metrics]) for d in data["total_conversations"]]
            r.append(mean_score_per_dialog)
    df = pd.DataFrame(r) # [n X D]
    sorted_mean_score_overfiles = df.mean(axis=0).sort_values(ascending=False)
    index_best_k = sorted_mean_score_overfiles[:k_positive].index if k_positive else []
    index_worst_k = sorted_mean_score_overfiles[-k_negative:].index if k_negative else []
    print(len(index_best_k), len(index_worst_k))
    # Collect top k cases
    eval_data = {}
    for i in range(len(json_files)):
        with open(f"result/{json_files[i]}", "r") as f:
            data = json.load(f)
            best = [data["total_conversations"][j] for j in index_best_k]
            worst = [data["total_conversations"][j] for j in index_worst_k]
            eval_data[json_files[i]] = best + worst
    with open(fname, 'w') as fw:
        json.dump(eval_data, fw, indent=4)
                    
if __name__ == "__main__":
    json_files = [
        'T-deepseek-llm_L-deepseek-llm_E-deepseek-llm_11246257_corrected.json',
        'T-qwen2_L-qwen2_E-qwen2_11236934_corrected.json',
        'T-gemma_L-gemma_E-gemma_11269574_corrected.json',
        'T-olmo2_L-olmo2_E-olmo2_11370185_corrected.json',
        'T-openchat_L-openchat_E-openchat_11297606_corrected.json',
        'T-llama3_L-llama3_E-llama3_11232754_corrected.json',
        'T-phi4_L-phi4_E-phi4_11269383_corrected.json',
    ]
    select_cases(json_files, k_positive=25, k_negative=0, fname='case_study/human_eval_conversation_mono_p25.json')
    select_cases(json_files, k_positive=0, k_negative=25, fname='case_study/human_eval_conversation_mono_n25.json')