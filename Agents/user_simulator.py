from langchain.memory import ConversationBufferMemory
from query_generator import generate_learning_queries
from web_scraper import retrieve_knowledge_from_wikihow

# Memory object
memory = ConversationBufferMemory()

def recall_memory():
    """Recall stored conversations or information."""
    return memory.load_memory_variables()

def ask_follow_up_questions(steps):
    """Simulate follow-up questions for unclear steps."""
    for i, step in enumerate(steps, 1):
        print(f"Step {i}: {step}")
        clarification_prompt = f"What additional details can you provide about this step: {step}?"
        detailed_explanation = llm(clarification_prompt)
        print(f"Clarification for Step {i}: {detailed_explanation}\n")

def simulate_application_scenario(steps):
    """Simulate applying the learned steps in a scenario."""
    for i, step in enumerate(steps, 1):
        scenario_prompt = f"""
        Imagine you are performing the following step in real life: {step}.
        Describe what might happen next, and how to proceed.
        """
        result = llm(scenario_prompt)
        print(f"Step {i} scenario result: {result}\n")

def user_simulator(topic):
    """Simulate a user learning and applying knowledge on a topic."""
    print(f"Topic: {topic}")

    # Step 1: Generate Questions
    print("\nGenerating questions...\n")
    questions = generate_learning_queries(topic)
    for question in questions:
        print(f"Question: {question}")

    # Step 2: Retrieve Knowledge
    print("\nRetrieving knowledge from Wikihow...\n")
    content, error = retrieve_knowledge_from_wikihow(topic)
    if error:
        print(error)
        return

    steps = content.get("steps", [])
    tips = content.get("tips", [])
    warnings = content.get("warnings", [])
    thingsyoullneed = content.get("thingsyoullneed", [])
    qs = content.get("qa", [])

    print("Steps:")
    for i, step in enumerate(steps, 1):
        print(f"Step {i}: {step}")

    print("\nTips:")
    for tip in tips:
        print(f"- {tip}")

    print("\nWarnings:")
    for warning in warnings:
        print(f"- {warning}")

    # Step 3: Ask Follow-Up Questions
    print("\nAsking clarifying questions...\n")
    ask_follow_up_questions(steps)

    # Step 4: Simulate Application Scenarios
    print("\nSimulating application scenarios...\n")
    simulate_application_scenario(steps)

    # Step 5: Collect things you need
    # section thingsyoullneed sticky

    # Optional: Store in memory
    memory.save_context({"user_query": topic}, {"retrieved_content": content})
    print("\nStored conversation in memory.")

if __name__ == "__main__":
    topic = "how to change a tire"
    user_simulator(topic)