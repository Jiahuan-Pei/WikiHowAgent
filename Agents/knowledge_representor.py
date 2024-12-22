from langchain.schema import Document

class KnowledgeBase:
    """Store and organize knowledge from Wikihow and LangChain."""
    def __init__(self):
        self.data = {}

    def add_topic(self, topic, content):
        """Add content related to a topic."""
        document = Document(page_content=str(content), metadata={"topic": topic})
        self.data[topic] = document

    def get_topic_content(self, topic):
        """Retrieve content for a specific topic."""
        return self.data.get(topic, "No information available for this topic.")

    def list_topics(self):
        """List all stored topics."""
        return list(self.data.keys())
