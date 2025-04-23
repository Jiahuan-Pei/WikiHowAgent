import sys, json
from statistics import mean, pstdev

# Define full set of metrics
full_metrics = {
    "Question Ratio": 0.0,
    "Completion Achieved": 0,
    "Diversity Score": 0.0,
    "Clarity": 0,
    "Engagement": 0,
    "Coherence": 0,
    "Depth": 0,
    "Relevance": 0,
    "Progress": 0,
    "Naturalness": 0,
    "Truthfulness": 0,
    "BLEU": 0.0,
    "METEOR": 0.0,
    "BERTScore": 0.0,
    "ROUGE": 0.0
}

def correct_result(fpath):
    # Load the partial input JSON file
    with open(fpath, "r") as f:
        data = json.load(f)

    # Function to parse scores from a string response
    def parse_score_string(score_str):
        result = {}
        parts = score_str.replace("\n", ",").replace("**", "").replace("/5", "").split(",")
        for part in parts:
            if ":" in part:
                try:
                    k, v = part.split(":")
                    result[k.strip()] = float(v.strip())
                except ValueError:
                    continue
        return result

    # Update each evaluation dict with missing metrics
    updated_conversations = []
    metrics_accumulator = {k: [] for k in full_metrics.keys()}

    for conv in data["total_conversations"]:
        eval_data = conv.get("evaluation", {})
        
        # Extract from string responses if missing
        if "llm_scores_response" in eval_data:
            llm_parsed = parse_score_string(eval_data["llm_scores_response"])
            eval_data.update(llm_parsed)
        if "ref_scores_response" in eval_data:
            ref_parsed = parse_score_string(eval_data["ref_scores_response"])
            eval_data.update(ref_parsed)

        # Fill missing metrics with 0
        for metric in full_metrics:
            if metric not in eval_data:
                eval_data[metric] = full_metrics[metric]
            metrics_accumulator[metric].append(eval_data[metric])

        conv["evaluation"] = eval_data
        updated_conversations.append(conv)

    # Recalculate total_evaluation as [mean, std_dev]
    total_evaluation = {
        metric: [mean(values), pstdev(values)] for metric, values in metrics_accumulator.items()
    }

    # Save the corrected output
    corrected_data = data
    corrected_data["total_conversations"] = updated_conversations
    corrected_data["total_evaluation"] = total_evaluation

    with open(f"{fpath.split('.')[0]}_corrected.json", "w") as f:
        json.dump(corrected_data, f, indent=4)


if __name__ == "__main__":
    if len(sys.argv)==2:
        """
        Usage example:
        python utils/post_result_correction.py result/T-gemma_L-gemma_E-gemma_11269574.json
        """
        correct_result(fpath=sys.argv[1])
    elif len(sys.argv)==1:
        # fpath = "result/T-deepseek-llm_L-deepseek-llm_E-deepseek-llm_012.json"
        fpaths = [
            'result/T-llama3_L-llama3_E-llama3_11232754.json'
        ]
        for fpath in fpaths:
            correct_result(fpath=fpath)