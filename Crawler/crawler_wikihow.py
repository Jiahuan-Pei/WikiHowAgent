import os
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from urllib.parse import urljoin, urlparse

# Base URL for WikiHow categories
BASE_URL = 'https://www.wikihow.com'
VIDEO_URL = f'{BASE_URL}/video'
IMAGE_URL = f'{BASE_URL}/images'

# Headers to simulate a browser request
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
}

def get_category_links():
    """Get links to all categories of tutorials."""
    url = f'{BASE_URL}/Special:CategoryListing#AllCategories'
    
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all category links
    categories = [e for e in soup.find_all('a') if 'Category:' in e.get('href', '')]

    category_links = []
    for cat in categories:
        link = BASE_URL + cat.get('href')
        category_name = cat.get_text(strip=True).replace(' ', '-')
        category_links.append((category_name, link))
    
    return category_links


def remove_javascript_img_only(soup):
    # Remove all <script> tags
    for script_tag in soup.find_all('script'):
        script_tag.decompose()

    # Remove inline JavaScript attributes (e.g., onclick, onload, etc.)
    js_attributes = ["onclick", "onload", "onerror", "onmouseover", "onmouseout", "onkeydown", "onkeyup", "onfocus", "onblur"]

    # Iterate through all tags in the document
    for tag in soup.find_all(True):  # True matches all tags
        for attr in js_attributes:
            if attr in tag.attrs:  # Check if the tag has a JavaScript attribute
                del tag.attrs[attr]  # Remove the attribute

    # Reassign 'src' with 'data-src' for all <img> elements
    for img in soup.find_all('img'):
        if img.get('data-src'):  # Check if 'data-src' exists
            img['src'] = img['data-src']
    return soup
    

def download_one_tutorial(title, url, category_name, output_root_dir):
    """
    Download a single tutorial, save the HTML content, and download all images from 'data-src'.
    
    Args:
        title (str): Title of the tutorial.
        url (str): URL of the tutorial.
        category_name (str): Name of the category the tutorial belongs to.
        output_root_dir (str): Root directory to save the tutorial and images.
    """
    try:
        # Save tutorials by category
        output_dir = os.path.join(output_root_dir, category_name, title)
        img_dir = os.path.join(output_dir, 'images')
        video_dir = os.path.join(output_dir, 'video')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(img_dir)
            os.makedirs(video_dir)

        # Fetch the tutorial page
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()

        # Parse the HTML to find images with 'data-src'
        soup = BeautifulSoup(response.content, 'html.parser')                   

        txt_filename = os.path.join(output_dir, f"{url.split('/')[-1]}.txt")
        txt_fw = open(txt_filename, 'w', encoding='utf-8')

        # Get sections with the steps containing step number
        sections = soup.select('[class*="section steps"]:not(.hide_step_numbers)')
        for section in sections:
            # Save textual instructions
            section_headline = section.find('div', {'class': 'headline_info'}).get_text().strip().strip('\n')
            step_nums = [t.get_text().strip() for t in section.find_all('div', {'class': 'step_num'})]
            step_instructions = [t.get_text().strip() for t in section.find_all('div', {'class': 'step'})]
            txt_fw.write(section_headline+'\n\n')
            for i, step in zip(step_nums, step_instructions):
                txt_fw.write(f'Step {i} {step}\n')
            # Save images
            images = section.find_all('img', {'data-src': True})
            for img in images:
                img_url = img['data-src'] # os.path.join(IMAGE_URL, img['data-src']) if BASE_URL not in img['data-src'] else img['data-src']
                download_image(img_url, img_dir)
            # Save videos if no images
            videos = section.find_all('video', {'data-src': True})
            for video in videos:
                video_url = VIDEO_URL + video['data-src'] # if BASE_URL not in video['data-src'] else video['data-src']
                print('*'*10,video_url)
                download_image(video_url, video_dir)
        
        # Check if the image and video folders are empty
        if not os.listdir(video_dir): 
            os.rmdir(video_dir)
            # Image only, then remove js to show images
            soup = remove_javascript_img_only(soup)

        if not os.listdir(img_dir):
            os.rmdir(img_dir)

        # Save the modified HTML into a new file that can show the images
        html_filename = os.path.join(output_dir, f"{url.split('/')[-1]}.html")
        with open(html_filename, 'w', encoding='utf-8') as file:
            file.write(soup.prettify()) 
        print(f"Saved HTML: {category_name}/{html_filename}")          
        # Respectful crawling: Pause between requests
        time.sleep(1)
    except Exception as e:
        print(f"Failed to save {url}: {e}")


def download_image(img_url, output_dir):
    """
    Download and save an image to the specified directory.
    
    Args:
        img_url (str): URL of the image to download.
        output_dir (str): Directory to save the image.
    """
    try:
        # Extract the image name from the URL
        # img_name = img_url #.split("/")[-1]
        img_name = os.path.basename(urlparse(img_url).path)
        img_path = os.path.join(output_dir, str(img_name))

        # Download the image
        response = requests.get(img_url, headers=HEADERS, stream=True)
        response.raise_for_status()
        with open(img_path, 'wb') as img_file:
            for chunk in response.iter_content(1024):
                img_file.write(chunk)

        print(f"Downloaded image: {img_path}")
    except Exception as e:
        print(f"Failed to download image {img_url}: {e}")

def get_tutorials_from_category(category_url):
    """Get tutorials for a specific category."""
    tutorials = []
    response = requests.get(category_url, headers=HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Get all tutorial links on the category page
    articles = [e for e in soup.find_all('a') if BASE_URL in e.get('href', '')]
    
    for article in articles:
        tutorial_title = article.get_text(strip=True)
        tutorial_link = article.get('href')
        if 'How to' in tutorial_title:
            tutorial_title = tutorial_title.replace('How to', '')
            tutorials.append((tutorial_title, tutorial_link))        
    return tutorials

def crawl_wikihow(output_root_dir='./data/wikihow'):
    """Main crawler function."""   
    category_links = get_category_links()
    print(f'#Category: {len(category_links)}')
    all_tutorials = []

    print(f"Found {len(category_links)} categories. Crawling tutorials...")

    for category_name, category_url in category_links:
        print(f"Fetching tutorials from category: {category_name} ({category_url})")
        tutorials = get_tutorials_from_category(category_url)

        # For each tutorial, append category, title, and link
        for tutorial in tutorials:
            title, url = tutorial[0], tutorial[1]
            all_tutorials.append({
                'Category': category_name,
                'Title': title,
                'Link': url
            })
            # Save each tutortial to html         
            download_one_tutorial(title, url, category_name, output_root_dir) 

    # Save tutorials to CSV
    df = pd.DataFrame(all_tutorials)
    df.to_csv('./data/wikihow_tutorials.csv', index=False)
    print(f"Saved {len(all_tutorials)} tutorials to wikihow_tutorials.csv.")

if __name__ == '__main__':
    crawl_wikihow()

