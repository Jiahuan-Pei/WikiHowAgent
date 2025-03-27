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
    
    # # Save the entire HTML content to a file for later understanding
    # with open('wikihow_search_page.html', 'w', encoding='utf-8') as html_file:
    #     html_file.write(response.text)  # Save the HTML content

    soup = BeautifulSoup(response.text, 'html.parser')

    doc_title = soup.select(".title_sm")[0].get_text()
    doc_intro = soup.select(".mf-section-0")[0].get_text()
    steps = [step.get_text(strip=True) for step in soup.select(".step")]
    qa_items = []
    qa_section = soup.select_one('div[class*="section_text"][id="qa"]')
    if qa_section:
        for item in qa_section.find_all('li'):
            question = 'Q: ' + item.select_one('div[class*="qa_q_txt"]').get_text(strip=True) + '\n'
            answer = 'A: ' + item.select_one('div[class*="qa_answer answe"]').get_text(strip=True)
            qa_items.extend([question, answer])
    else:
        qa_items = []    
    tips = [tip.get_text(strip=True) for tip in soup.select(".tip")]
    warnings = [warning.get_text(strip=True) for warning in soup.select(".warning")]
    return {
        "title": doc_title,
        "url": article_url,
        "doc_intro": doc_intro,
        "steps": steps,
        "qa": qa_items,
        "tips": tips,
        "warnings": warnings
    }

# def retrieve_knowledge_from_wikihow(topic):
#     """Simulate knowledge retrieval by scraping Wikihow."""
#     article_links = fetch_wikihow_article_links(topic)
#     if not article_links:
#         return None, "No articles found."
#     content = extract_wikihow_content(article_links[0])
#     return content, None

# from pywikihow import HowTo

# how_to = HowTo("https://www.wikihow.com/Train-a-Dog")

# data = how_to.as_dict()

# print(how_to.url)
# print(how_to.title)
# print(how_to.intro)
# print(how_to.n_steps)
# print(how_to.summary)

# first_step = how_to.steps[0]
# first_step.print()
# data = first_step.as_dict()

# how_to.print(extended=True)
