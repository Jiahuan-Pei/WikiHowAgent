import json

metrics = ["Clarity", "Engagement", "Coherence", "Depth",
           "Relevance", "Progress", "Naturalness", "Truthfulness"]

def select_cases(json_files, n_positive, n_negative):
    for i in range(len(json_files)):
        with open(json_files[i], "r") as f:
            data = json.load(f)
            data["total_conversations"]["evaluation"]

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
    select_cases(json_files, 50, 50)