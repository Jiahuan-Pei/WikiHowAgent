from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.util import setup_llm_and_embeddings
import re


# Learner Agent
class LearnerLLMAgent:
    def __init__(self, config):
        super().__init__()
        self.llm, _, self.config = setup_llm_and_embeddings(config)

        self.prompt_template = PromptTemplate(
            input_variables=["instruction"],
            template=(
                "You are a 🤔 curious student engaged in a step-by-step learning conversation with a teacher.\n\n"
                "🎯 Your role:\n"
                "- Carefully read and understand the teacher’s instructions.\n"
                "- Respond naturally, politely, and concisely.\n"
                "- If the instruction is **unclear**, ask a **brief and specific** question to clarify it.\n"
                "- If the instruction is **clear**, acknowledge it briefly and politely ask to move on to the next step.\n"
                "- If the teacher opens with a 'BEGIN' message, just acknowledge it and do **not** ask questions.\n"
                "- If the teacher says **'FINISHED'** or indicates that the tutorial is over, thank them politely and say nothing more.\n\n"
                "📝 Respond in character as a student — no system-level messages or meta-comments. Keep responses short and focused.\n"
                "Teacher: {instruction}\n"
                "Learner:"
            ),
        )
        
        self.chain = LLMChain(prompt=self.prompt_template, llm=self.llm, verbose=True)
        # self.chain =  RunnableSequence(self.prompt_template, self.llm)

    def respond(self, data: dict) -> str:
        try:
            learner_response = self.chain.invoke({"instruction": data["instruction"]})['text']
            # learner_response = learner_response.encode('utf-8').decode('unicode_escape')
            # Removing non-ASCII characters using regex
            # learner_response = re.sub(r'[^\x00-\x7F]+', '', learner_response)

        except:
            print('Learner:INVOKE:ERR', data, 'Please check if you have the current llm or your API key is valid.')

        print('<<🤔 Learner>>', learner_response)
        return learner_response
    
    def ask(self, data: dict) -> str:
        return
    
    def feedback(self, data: dict) -> str:
        return    