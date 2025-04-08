from langchain.prompts import PromptTemplate
from utils.util import setup_llm_and_embeddings
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, Literal, List

class LLMDecisionAgent:
    """LLM-powered agent for adaptive dialogue management."""
    def __init__(self, config):
        self.llm, _, self.config = setup_llm_and_embeddings(config)
        self.prompt_template = PromptTemplate(
            input_variables=["current_step_index", "needs_clarification", "conversation_text"],
            template=(
                "You are an intelligent tutor managing a structured learning dialogue. Your goal is to decide the next action based on the conversation history.\n\n"
                "🎯 Your role:\n"
                "- Alternate between 'teacher' and 'learner' based on the last speaker in the conversation.\n"
                "- If the last speaker was the 'teacher' and they said 'FINISHED', return '__end__'.\n\n"
                "ONLY alternate turn without self-loop: teacher <-> learner.\n"
                "Respond ONLY with one of the strings: 'teacher', 'learner', '__end__'. No explanations.\n"
                "Current Step: {current_step_index}\n"
                "Needs Clarification: {needs_clarification}\n"
                "Conversation History:\n"
                "{conversation_text}\n\n"
                "What is the next action?"
            ),
        )
        self.chain = LLMChain(prompt=self.prompt_template, llm=self.llm, verbose=True)
    
    def decide_next_step(self, state: Dict) -> Literal["teacher", "learner", "__end__"]:
        messages = state["messages"]
        conversation_text = "\n".join([
            f"{'Teacher' if isinstance(msg, AIMessage) else 'Learner'}: {msg.content}"
            for msg in messages
        ])
        data = {
            'current_step_index': state["current_step_index"],
            'needs_clarification': state["needs_clarification"],
            'conversation_text': conversation_text
        }
        decision = self.chain.invoke(data)['text'].strip().lower().replace("'", "")
        return decision if decision in {"teacher", "learner", "__end__"} else "teacher"