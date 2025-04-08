import re
import json
from pprint import pprint
from collections import OrderedDict
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import evaluate
from nltk.util import ngrams
import os, sys

# Custom utility imports
from utils.util import setup_llm_and_embeddings, load_yaml
from utils.setup_logger import setup_logger
logger = setup_logger() # Get the current script name

class ConversationEvaluator:
    def __init__(self, config):
        self.llm, self.embeddings, self.config = setup_llm_and_embeddings(config)
        with open(self.config['params']['rubric_file'], 'r') as fr:
            self.rubrics = json.load(fr)
        # Load BERT-based model for semantic similarity
        self.bert_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Load evaluation metrics from Hugging Face's evaluate library
        self.bleu = evaluate.load("bleu")
        self.meteor = evaluate.load("meteor")
        self.rouge = evaluate.load("rouge")
    
    # Question Ratio
    def calculate_question_ratio(self, conversation):
        """Calculates the ratio of learner messages that contain questions.

        Args:
            conversation: List of conversation lines
            
        Returns:
            float: Ratio of learner messages containing questions (0.0 to 1.0)
        """
        learner_messages = [line for line in conversation if line.startswith("Learner:")]
        if not learner_messages:  # Avoid division by zero
            return 0.0
        question_count = sum(1 for msg in learner_messages if "?" in msg)
        return round(question_count / len(learner_messages), 4)

    # Task Completion
    def check_completion(self, conversation):
        """Checks if the conversation ends with the tutorial completion message.
        Returns 1 if true, 0 otherwise."""
        return 1 if any("FINISHED" in line or "The tutorial is now complete" in line for line in conversation) else 0
    
    # Learner Response Diversity
    def check_diversity(self, conversation):
        """Measures linguistic diversity based on unique sentence structures."""
        learner_responses = [line for line in conversation]
        unique_responses = set(learner_responses)
        return round(len(unique_responses) / max(1, len(learner_responses)), 4)  # Avoid division by zero
    
    def compute_ngram_diversity(self, responses, n=2):
        """Computes n-gram diversity score (bigram, trigram, etc.)
        Score	Interpretation
        0.0 - 0.3	Highly repetitive phrases
        0.3 - 0.6	Some phrase variation, but noticeable repetition
        0.6 - 0.9	Good diversity, minimal repeated phrases
        0.9 - 1.0	Very rich vocabulary, almost no repetition
        """
        ngram_list = []
        for response in responses:
            words = response.split()
            ngram_list.extend(list(ngrams(words, n)))  # Extract n-grams
        
        total_ngrams = len(ngram_list)
        unique_ngrams = len(set(ngram_list))
        
        return round(unique_ngrams / max(1, total_ngrams), 4)  # Avoid division by zero
    
    # Revised BLEU Score (n-gram precision-based metric)
    def compute_bleu(self, references, generations):
        """
        Computes corpus-level or document-level BLEU score.
        
        If the lengths of references and generations differ, the function assumes
        that the entire list should be treated as a single document. In that case,
        it joins all generations into one document and—if there are multiple 
        references per generation—it combines each reference candidate into a full
        document, then computes a document-level BLEU score.
        
        :param references: List of reference texts. Each element can be a string or a list of strings 
                        (if multiple references per generation are provided).
        :param generations: List of generated texts.
        :return: BLEU score rounded to 4 decimal places.
        """
        if not references or not generations:
            logger.error("Error: Empty reference or generated list")
            return 0.0

        # If references are simple strings, wrap them to support multiple references per generation.
        if isinstance(references[0], str):
            references = [[ref] for ref in references]

        try:
            # Check if the number of reference items matches the number of generation items.
            if len(references) == len(generations):
                # Standard corpus-level BLEU computation.
                result = self.bleu.compute(predictions=generations, references=references)
                return round(result["bleu"], 4)
            else:
                # Document-level BLEU: combine all generations into one document.
                combined_generation = " ".join(generations)
                
                # For references, combine each reference candidate (assumed consistent across sentences)
                # into one document.
                # Determine how many reference candidates there are from the first sentence.
                num_candidates = len(references[0])
                combined_references = []
                for i in range(num_candidates):
                    candidate_sentences = []
                    for ref_list in references:
                        if i < len(ref_list):
                            candidate_sentences.append(ref_list[i])
                    # Join all sentences for this candidate into one document.
                    combined_references.append(" ".join(candidate_sentences))
                
                # Compute document-level BLEU on a single prediction with multiple reference documents.
                result = self.bleu.compute(predictions=[combined_generation], references=[combined_references])
                return round(result["bleu"], 4)
        except Exception as e:
            logger.error(f"BLEU computation failed: {e}")
            return 0.0
        

    # Instructional accuracy: ROUGE (useful for summarization-based tasks)
    def compute_rouge(self, reference, generated):
        """Computes ROUGE score using Hugging Face evaluate."""
        rouge_scores = self.rouge.compute(predictions=[generated], references=[reference])['rouge1']
        return {"ROUGE":round(float(rouge_scores), 4)}

    # METEOR Score (Recall-based metric, considers synonyms & stemming)
    def compute_meteor(self, reference, generated):
        """Computes METEOR score using Hugging Face evaluate."""
        meteor_scores = self.meteor.compute(predictions=[generated], references=[reference])['meteor']
        return round(float(meteor_scores), 4)
        
    # Semantic similarity: BERTScore (context-aware similarity, recall-based matching)
    def compute_bert_score(self, reference, generated):
        """Computes semantic similarity using BERT embeddings."""
        ref_embedding = self.bert_model.encode(reference, convert_to_tensor=True)
        gen_embedding = self.bert_model.encode(generated, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(ref_embedding, gen_embedding).item()
        return round(similarity, 4)

    # Instruction-tutorial consistency
    def check_factual_consistency(self, reference, generated):
        """Uses LLM to check the generated response against the tutorial reference."""
        rubric_prompt = "\n".join([f"- {key} ({value['question']})" for key, value in self.rubrics.items() if self.rubrics[key]['reference'] == 'tutorial'])
        format_prompt = ", ".join([f"{key}: X" for key in self.rubrics.keys() if self.rubrics[key]['reference'] == 'tutorial'])
        prompt = f"""
        Evaluate the following generated response based on the reference:
        Reference: {reference}
        Generated: {generated}
        
        Score from 1 to 5:
        {rubric_prompt}
        Provide scores in this format: {format_prompt}
        """
        response = self.llm.invoke(prompt).content
        # logger.info(response)
        matches = re.findall(r'(\w+):\s*(\d+)\s*', response, re.IGNORECASE)
        scores = {}
        for key, value in matches:
            if key in self.rubrics.keys():
                scores[key.capitalize()] = int(value)
        return scores, response

    def evaluate_with_reference(self, conversation, tutorial, role="Teacher"):
        reference = '\n'.join(tutorial)
        generated = '\n'.join([line for line in conversation if f"{role}:" in line])
        rouge_scores = self.compute_rouge(reference, generated)
        ref_llm_scores, response = self.check_factual_consistency(reference, generated)
        scores = {
            **ref_llm_scores,
            'BLEU': self.compute_bleu(reference, generated),
            'METEOR': self.compute_meteor(reference, generated),
            'BERTScore': self.compute_bert_score(reference, generated),
            **rouge_scores
        }
        return scores, response

    def evaluate_with_llm(self, conversation):
        """Uses an LLM to rate the conversation on multiple rubrics."""
        # Build the prompt using the rubrics dictionary
        rubric_prompt = "\n".join([f"- {key} ({value['question']})" for key, value in self.rubrics.items() if self.rubrics[key]['reference'] == 'no'])
        format_prompt = ", ".join([f"{key}: X" for key in self.rubrics.keys() if self.rubrics[key]['reference'] == 'no'])
        prompt = f"""
        Evaluate the following teacher-learner conversation:
        {conversation}
        
        Score from 1 to 5:
        {rubric_prompt}
        Provide scores in this format: {format_prompt}
        """

        response = self.llm.invoke(prompt).content
        # logger.info(response)
        matches = re.findall(r'(\w+):\s*(\d+)\s*', response, re.IGNORECASE)
        scores = {}
        for key, value in matches:
            if key in self.rubrics.keys():
                scores[key.capitalize()] = int(value)
        return scores, response
    
    def evaluate(self, conversation: List[str], tutorial: List[str]=None):
        num_questions = self.calculate_question_ratio(conversation)
        completed = self.check_completion(conversation)
        diversity_score = self.check_diversity(conversation)
        ngram_diversity_score = self.compute_ngram_diversity(conversation)
        llm_scores, llm_scores_response = self.evaluate_with_llm(conversation)
        if tutorial and conversation:
            ref_scores, ref_scores_response = self.evaluate_with_reference(conversation, tutorial)
            return OrderedDict({
                "Question Ratio": num_questions,
                "Completion Achieved": completed,
                "Diversity Score": round(ngram_diversity_score, 4),
                **llm_scores,  # Unpack the LLM scores
                **ref_scores,
                "llm_scores_response": llm_scores_response,
                "ref_scores_response": ref_scores_response
            })
        else:
            return OrderedDict({
                "Question Ratio": num_questions,
                "Completion Achieved": completed,
                "Diversity Score": round(diversity_score, 4),
                **llm_scores,  # Unpack the LLM scores
                "llm_scores_response": llm_scores_response,
            })

if __name__ == "__main__":
    # Example Usage:
    tutorial = [
        "Fill a container with salt.",
        "Squeeze a little tempera paint into the salt.",
        "Mix with a spoon until evenly distributed.",
        "Let it dry overnight.",
        "Test before using in crafts."
    ]
    conversation = [
        "Teacher: Step 1: Fill a container with salt.",
        "Learner: Should I use fine or coarse salt?",
        "Teacher: Use fine salt for better mixing.",
        "Learner: Got it. Please proceed to the next step.",
        "Teacher: Step 2: Squeeze a little tempera paint into the salt.",
        "Learner: I understand. Please proceed to the next step.",
        "Teacher: Step FINAL: The tutorial is now complete. FINISHED.",
        "Learner: Thank you!"
    ]
    config_file='conf/ollma-llama3.yaml'
    config = load_yaml(config_file)
    evaluator = ConversationEvaluator(config)
    results = evaluator.evaluate(conversation, tutorial)
    # pprint(results, indent=4)
    # logger.info(results)
