from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def interaction_manager_with_langchain(topic):
    """Simulate interaction with LangChain to learn a topic."""
    # Generate queries
    print("Generating queries...")
    queries = generate_queries_with_langchain(topic)
    for query in queries:
        print(f"- {query}")

    # Scrape Wikihow
    print("\nFetching Wikihow articles...")
    article_links = fetch_wikihow_article_links(topic)
    if not article_links:
        print("No articles found.")
        return

    print(f"Found {len(article_links)} article(s). Fetching the first one...")
    content = extract_wikihow_content(article_links[0])
    if isinstance(content, str):  # Error message
        print(content)
        return

    print("\nSteps:")
    for i, step in enumerate(content["steps"], 1):
        print(f"Step {i}: {step}")

    # Clarify with LangChain
    print("\nClarifying steps with LangChain...")
    for i, step in enumerate(content["steps"], 1):
        prompt_template = """
        You are a helpful assistant. Explain this step in detail:
        Step {step}.
        """
        prompt = PromptTemplate(input_variables=["step"], template=prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        detailed_explanation = chain.run(step=step)
        print(f"Step {i} detailed explanation: {detailed_explanation}\n")
