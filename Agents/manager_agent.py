from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, Literal, List
from utils.util import setup_llm_and_embeddings

class ManagerLLMAgent:
    """LLM-powered agent for adaptive dialogue management."""
    def __init__(self, config):
        super().__init__()
        self.llm, _, self.config = setup_llm_and_embeddings(config)
        self.prompt_template = PromptTemplate(
            input_variables=["current_step_index", "total_step_length", "needs_clarification", "finished", "last_speaker"],
            template=(
                "You are the conversation manager in a tutorial session between a teacher and a learner.\n"
                "Your job is to decide who should speak next: the 'teacher', the 'learner', or if the conversation should '__end__'.\n\n"
                "Here are the guidelines:\n"
                "- If the learner is confused or asking questions, the teacher should speak.\n"
                "- If the teacher has just spoken (last utterance is from teacher), the learner should speak next, and vice versa.\n"
                "- If the tutorial is finished or the teacher says 'FINISHED', end the conversation.\n"
                "- The tutorial step index is currently at {current_step_index} out of {total_step_length}.\n"
                "- The flag 'needs_clarification' is {needs_clarification}\n"
                "- The flag 'finished' is {finished}.\n\n"
                "Here is the last speaker:\n"
                "{last_speaker}\n\n"
                "Who should speak next? Reply with only one of: teacher, learner, or __end__."
            ),
        )
        self.chain = LLMChain(prompt=self.prompt_template, llm=self.llm, verbose=True)
    
    def invoke(self, state: Dict) -> Literal["teacher", "learner", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        last_speaker = f"{"teacher" if isinstance(last_message, AIMessage) else "learner"}"

        # Inject derived or fallback values into state for consistent access
        derived_state = {
            "current_step_index": state["current_step_index"],
            "total_step_length": len(state.get("tutorial", [])),
            "needs_clarification": state["needs_clarification"],
            "finished": state["finished"],
            "last_speaker": last_speaker,
        }
        decision = self.chain.invoke(derived_state)['text'].strip().lower().replace("'", "")
        print("🧠 Current step index:", state["current_step_index"], "/", len(state["tutorial"]))
        print("📚 Total messages so far:", len(state["messages"]))
        print("❓ Needs clarification:", state["needs_clarification"])
        print("✅ Finished flag:", state["finished"])
        print("🧭 Decision:", decision)
        return decision if decision in {"teacher", "learner", "__end__"} else "teacher"