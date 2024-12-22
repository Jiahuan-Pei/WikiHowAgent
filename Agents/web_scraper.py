import requests
from bs4 import BeautifulSoup

def fetch_wikihow_article_links(topic):
    """Fetch article links from Wikihow based on a search query."""
    search_url = f"https://www.wikihow.com/wikiHowTo?search={topic}"
    response = requests.get(search_url)
    if response.status_code != 200:
        return f"Failed to fetch Wikihow search page. Status code: {response.status_code}"

    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.select(".result_link")
    return [article['href'] for article in articles] if articles else []

def extract_wikihow_content(article_url):
    """Extract steps, tips, and warnings from a Wikihow article."""
    response = requests.get(article_url)
    if response.status_code != 200:
        return f"Failed to fetch article. Status code: {response.status_code}"

    soup = BeautifulSoup(response.text, 'html.parser')
    steps = [step.get_text(strip=True) for step in soup.select(".step")]
    tips = [tip.get_text(strip=True) for tip in soup.select(".tip")]
    warnings = [warning.get_text(strip=True) for warning in soup.select(".warning")]
    return {
        "steps": steps,
        "tips": tips,
        "warnings": warnings
    }

def retrieve_knowledge_from_wikihow(topic):
    """Simulate knowledge retrieval by scraping Wikihow."""
    article_links = fetch_wikihow_article_links(topic)
    if not article_links:
        return None, "No articles found."
    content = extract_wikihow_content(article_links[0])
    return content, None
