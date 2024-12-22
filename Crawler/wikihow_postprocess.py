import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from bs4 import BeautifulSoup
import os
import json
import networkx as nx
from pyvis.network import Network
import colorsys

def load_wikihow_csv():
    # Load the CSV file
    csv_file = 'data/wikihow_tutorials.csv'
    df = pd.read_csv(csv_file)
    df.drop_duplicates(inplace=True)

    # 1. Data Inspection
    print("First few rows of data:")
    print(df.head())

    # 2. Check for any missing or duplicated data
    print("\nCheck for missing values:")
    print(df.isnull().sum())  # Check missing values
    print("\nCheck for duplicates:")
    print(df.duplicated().sum())  # Check for duplicates
    return df

# 3. Analyze distribution of tutorials per category
def plot_tasks_per_topic(df):
    category_counts = df['Category'].value_counts()
    print("\nNumber of Tasks per Topic:")
    print(category_counts)

    # Number of categories
    num_categories = len(category_counts)

    # Define angles for each bar (each category will get an equal angle)
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()

    # Create a figure with polar axes (this is necessary for a radial chart)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Bar widths based on the number of categories
    width = np.pi / 4  # Adjust the width of each bar for better separation

    # Normalize the category counts for color mapping
    norm = Normalize(vmin=category_counts.min(), vmax=category_counts.max())

    # Choose a colormap that goes from blue to red (coolwarm)
    cmap = cm.coolwarm  # This is a diverging colormap from blue to red


    # Create radial bars
    bars = ax.bar(angles, category_counts, width=0.08*width, bottom=0.0, color=[cmap(norm(count)) for count in category_counts])

    # Add labels to each bar
    ax.set_xticks(angles)  # Set category labels around the circle
    cate_nums = [f"[{i+1}]" for i, name in enumerate(category_counts.index)]
    ax.set_xticklabels([]*num_categories)

    num_label = []
    # Adjust label position closer to the circle using ax.text() for fine control
    for i, label in enumerate(category_counts.index):
        # print(f"[{i+1}] {label}")
        num_label.append((f"[{i+1}]", label, category_counts[label]))
        angle = angles[i]
        # Position the text closer to the circle by changing the radius (0.85 * max count in this case)
        ax.text(angle, 1.05 * category_counts.max(), f"[{i+1}]", ha='center', va='center', fontsize=6)
        
    # Add a title
    # ax.set_title('#Tasks per Topic', va='bottom', fontsize=14)

    # Customize the polar plot
    ax.set_ylim(0, category_counts.max() + 1)  # Adjust the radial limit

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'figure/tasks_per_topic.png')
    df_num_topic = pd.DataFrame(num_label, columns=['Number', 'Topic', 'Count'])
    # Write to CSV
    df_num_topic.to_csv('figure/tasks_per_topic.csv', index=False)
    # plt.show()

def plot_title_length(df):
    # 6. Additional Analysis: Length of Titles
    df['Title_Length'] = df['Title'].apply(len)
    plt.figure(figsize=(10, 6))
    df['Title_Length'].hist(bins=20, edgecolor='lightblue')
    # plt.title('Distribution of Title Lengths')
    plt.xlabel('Title Length')
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'figure/title_length.png')

def plot_title_length_by_topic(df):
    df['Title_Length'] = df['Title'].apply(len)
    # Calculate mean title length per category and sort
    mean_lengths = df.groupby('Category')['Title_Length'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    mean_lengths.plot(kind='bar', edgecolor='lightblue')
    plt.xlabel('Category')
    plt.ylabel('Average Title Length')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f'figure/title_length_by_topic.png')

def convert_html_to_md(mdir='./data/wikihow'):    
    markdown_docs = []
    for topic in os.listdir(mdir):
        topic_path = os.path.join(mdir, topic)
        if not os.path.isdir(topic_path):
            continue
        for task in os.listdir(topic_path):
            fpath = f'{mdir}/{topic}/{task}/{task}.md'
            print(fpath)
            task_path = os.path.join(topic_path, task)
            if not os.path.isdir(task_path):
                continue
            for file in os.listdir(task_path):             
                if file.endswith('.html'):
                    with open(os.path.join(mdir, topic, task, file), 'r') as f:
                        soup = BeautifulSoup(f, 'html.parser')
                        # Find all elements with class containing "section"
                        breadcrumb = soup.find(id='breadcrumb')
                        if breadcrumb:
                            categories = [c.get_text(strip=True) for c in breadcrumb.find_all('a') if c.get_text(strip=True)]
                        else:
                            categories = []
                            print(f"Warning: No breadcrumb found in {topic}/{task}")
                        title_element = soup.select_one('h1[class*="title"] a')
                        if title_element:
                            title = title_element.get_text(strip=True)
                        else:
                            # Try without the anchor tag in case title is direct text
                            title_element = soup.select_one('h1[class*="title"]')
                            if title_element:
                                title = title_element.get_text(strip=True)
                            else:
                                title = "Unknown Title"
                                print(f"Warning: No title found in {topic}/{task}")
                                continue # Skip the current unknow doc 
                                
                        intro = soup.find('div', id='mf-section-0').get_text(strip=True)
                        step_lists = soup.select('div[class*="section steps"]')
                        methods = []
                        for method in step_lists:
                            method_title = method.select_one('div[class*="headline_info"]').get_text(strip=True)
                            steps = []
                            step_section = method.select_one('div[class="section_text"]')
                            if step_section:
                                # Try different possible step structures
                                try:
                                    for step_num_elem, step_elem in zip(step_section.select('div[class="step_num"]'), step_section.select('div[class="step"]')):
                                        if step_num_elem and step_elem:
                                            step_num = step_num_elem.get_text(strip=True)
                                            step_text = step_elem.get_text(strip=True).replace('XResearch resource', '')
                                            steps.append(f"{step_num}. {step_text}")
                                except:
                                    print(f"Error in {topic}/{task}/{file}")
                                    exit()
                            methods.append([method_title] + steps)
                        qa_items = []
                        qa_section = soup.select_one('div[class*="section_text"][id="qa"]')
                        if qa_section:
                            for item in qa_section.find_all('li'):
                                question = 'Q: ' + item.select_one('div[class*="qa_q_txt"]').get_text(strip=True) + '\n'
                                answer = 'A: ' + item.select_one('div[class*="qa_answer answe"]').get_text(strip=True)
                                qa_items.extend([question, answer])
                        else:
                            qa_items = []
                        tips_section = soup.select_one('div[class*="section_text"][id="tips"]')
                        if tips_section:
                            tips = [t.get_text(strip=True) for t in tips_section.select('li')]
                        else:
                            tips = []
                        warnings_section = soup.select_one('div[class*="section_text"][id="warnings"]')
                        if warnings_section:
                            warnings = [w.get_text(strip=True) for w in warnings_section.select('li')]
                        else:
                            warnings = []
                        thingsyoullneed_section = soup.select_one('div[class*="section_text"][id="thingsyoullneed"]')
                        if thingsyoullneed_section:
                            # Find all method headlines in the "Things You'll Need" section
                            method_headlines = thingsyoullneed_section.select('div.headline_info h3')
                            things_needed = []
                            
                            for headline in method_headlines:
                                method_name = headline.get_text(strip=True)
                                # Find the ul that follows this headline
                                next_ul = headline.find_next('ul')
                                if next_ul:
                                    items = [item.get_text(strip=True) for item in next_ul.find_all('li')]
                                    things_needed.append((method_name, items))
                            
                            # Format the things needed section
                            if things_needed:
                                thingsyoullneed = []
                                for method_name, items in things_needed:
                                    thingsyoullneed.append(f"### {method_name}")
                                    thingsyoullneed.extend([f"* {item}" for item in items])
                        else:
                            thingsyoullneed = []
                        
                        references_section = soup.select_one('div[class*="section_text"][id="references"]')
                        if references_section:
                            references = [r.get_text(strip=True).replace('↑', '') for r in references_section.find_all('li')]
                        else:
                            references = []
                        # Combine text from all matching sections
                        markdown = f"""# {title}\n\n\n
## Category
{' >> '.join(categories)}\n\n\n
## Introduction
{intro}\n\n\n
## Methods
{'\n'.join([f"### {m[0]}\n" + '\n'.join([f"{step}" for step in m[1:]]) for m in methods])}\n\n\n
## Q&A
{'\n'.join([f"**Q:** {q[2:]}\n**A:** {a[2:]}\n" for q, a in zip(qa_items[::2], qa_items[1::2])])}\n\n\n
## Tips
{'\n'.join([f"* {t}" for t in tips])}\n\n\n
## Warnings
{'\n'.join([f"* {w}" for w in warnings])}\n\n\n
## Things You'll Need
{'\n'.join(thingsyoullneed)}\n\n\n
## References
{'\n'.join([f"[{i+1}] {r}\n" for i, r in enumerate(references)])}
"""
                        # print(plain_text)
                        # soup.find('div', {'class': 'content'}).decompose()
                        # fpath = f'{mdir}/{topic}/{task}.md'
                        with open(fpath, 'w') as f:
                            f.write(markdown)
                        
                        markdown_docs.append(markdown)
    return markdown_docs

def read_markdown_docs(mdir='./data/wikihow'):
    # Read all markdown docs from directory
    markdown_docs = []
    for topic in os.listdir(mdir):
        topic_path = os.path.join(mdir, topic)
        if not os.path.isdir(topic_path):
            continue
        for task in os.listdir(topic_path):
            md_file = os.path.join(topic_path, task, f"{task}.md")
            if os.path.exists(md_file):
                with open(md_file, 'r') as f:
                    markdown_docs.append(f.read())
    return markdown_docs

def calculate_data_statistics_of_markdown_docs():
    # Read all markdown_docs 
    markdown_docs = read_markdown_docs()

    # Initialize statistics dictionaries
    stats = {
        'document_tokens': [],
        'category_tokens': {},
        'method_tokens': [],
        'step_tokens': [],
        'qa_tokens': [],
        'tip_tokens': [],
        'warning_tokens': [],
        'num_reference': [],
        'steps_per_method': [],
        'methods_per_doc': [],
        'qas_per_doc': [],
    }
    
    for doc in markdown_docs:
        # Split document into sections
        sections = doc.split('\n\n\n\n')
        doc_tokens = len(doc.split())
        stats['document_tokens'].append(doc_tokens)
        list_of_categories = []
        method_count = 0  # Count methods in current doc
        current_method_steps = 0  # Count steps in current method
        qa_count = 0 # Count QA in current doc
        for section in sections:
            section = section.lstrip()
            try:
                if section.startswith('## Category'):
                    # Process categories
                    categories = section.split('\n')[1].split(' >> ')
                    list_of_categories.append(categories)
                    for category in categories:
                        stats['category_tokens'][category] = stats['category_tokens'].get(category, 0) + len(category.split())
                
                elif section.startswith('## Methods'):
                    # Process methods and steps
                    method_lines = section.split('\n')[1:]
                    current_method_tokens = 0
                    
                    for line in method_lines:
                        if line.startswith('###'):
                            if current_method_tokens > 0:
                                stats['method_tokens'].append(current_method_tokens)
                            current_method_tokens = len(line.split())
                            # New method found
                            if current_method_steps > 0:
                                stats['steps_per_method'].append(current_method_steps)
                            current_method_steps = 0
                            method_count += 1
                        elif line.strip() and line[0].isdigit():
                            step_tokens = len(line.split())
                            current_method_tokens += step_tokens
                            stats['step_tokens'].append(step_tokens)
                            # New method found
                            current_method_steps += 1
                    
                    if current_method_tokens > 0:
                        stats['method_tokens'].append(current_method_tokens)
                    
                    # Don't forget to append the last method's step count
                    if current_method_steps > 0:
                        stats['steps_per_method'].append(current_method_steps)
                
                elif section.startswith('## Q&A'):
                    # Process Q&A pairs
                    qa_lines = [line for line in section.split('\n') if line.startswith('**')]
                    
                    for i in range(0, len(qa_lines), 2):
                        if i + 1 < len(qa_lines):
                            qa_count +=1
                            qa_tokens = len(qa_lines[i].split()) + len(qa_lines[i+1].split())
                            stats['qa_tokens'].append(qa_tokens)
                
                elif section.startswith('## Tips'):
                    # Process tips
                    tip_lines = [line for line in section.split('\n') if line.startswith('*')]
                    # print("Tip lines found:", len(tip_lines))  # Debug print
                    # print("First few tip lines:", tip_lines[:2])  # Debug print
                    for tip in tip_lines:
                        stats['tip_tokens'].append(len(tip.split()))
                
                elif section.startswith('## Warnings'):
                    # Process warnings
                    warning_lines = [line for line in section.split('\n') if line.startswith('*')]
                    for warning in warning_lines:
                        stats['warning_tokens'].append(len(warning.split()))
                
                elif section.startswith('## References'):
                    # Process references
                    ref_lines = [line for line in section.split('\n') if line.startswith('[')]
                    stats['num_reference'].append(len(ref_lines))
            except Exception as e:
                print('Statistic ERR', e)
                
        # Moved outside the section loop - append method count once per document
        stats['methods_per_doc'].append(method_count)
        stats['qas_per_doc'].append(qa_count)
    
    # Calculate and print summary statistics
    print("\nDocument Statistics:")
    print(f"Total documents: {len(stats['document_tokens'])}")
    print(f"Average tokens per document: {np.mean(stats['document_tokens']):.2f}")
    
    # print("\nCategory Statistics:")
    # for category, tokens in stats['category_tokens'].items():
    #     print(f"{category}: {tokens} tokens")
    
    print("\nMethod Statistics:")
    print(f"Total methods: {len(stats['method_tokens'])}")
    print(f"Average tokens per method: {np.mean(stats['method_tokens']):.2f}")
    
    print("\nStep Statistics:")
    print(f"Total steps: {len(stats['step_tokens'])}")
    print(f"Average tokens per step: {np.mean(stats['step_tokens']):.2f}")
    
    print("\nQ&A Statistics:")
    print(f"Total Q&A pairs: {len(stats['qa_tokens'])}")
    print(f"Average tokens per Q&A: {np.mean(stats['qa_tokens']) if stats['qa_tokens'] else 0:.2f}")
    
    print("\nTip Statistics:")
    print(f"Total tips: {len(stats['tip_tokens'])}")
    print(f"Average tokens per tip: {np.mean(stats['tip_tokens']) if stats['tip_tokens'] else 0:.2f}")
    
    print("\nWarning Statistics:")
    print(f"Total warnings: {len(stats['warning_tokens'])}")
    print(f"Average tokens per warning: {np.mean(stats['warning_tokens']) if stats['warning_tokens'] else 0:.2f}")
    
    print("\nReference Statistics:")
    print(f"Total references: {len(stats['num_reference'])}")
    print(f"Average tokens per reference: {np.mean(stats['num_reference']) if stats['num_reference'] else 0:.2f}")

    print("\nMethod and Step Analysis:")
    print(f"Total number of methods across all docs: {sum(stats['methods_per_doc'])}")
    print(f"Average methods per document: {np.mean(stats['methods_per_doc']):.2f}")
    print(f"Total number of QAs across all docs: {sum(stats['qas_per_doc'])}")
    print(f"Average QAs per document: {np.mean(stats['qas_per_doc']):.2f}")
    print(f"Total number of steps across all docs: {sum(stats['steps_per_method'])}")
    print(f"Average steps per method: {np.mean(stats['steps_per_method']):.2f}")
    
    return stats

def build_up_knowledge_graph_interactive():
    try:
        # Create the output directory if it doesn't exist
        os.makedirs('figure', exist_ok=True)
        
        # Read all markdown_docs 
        markdown_docs = read_markdown_docs()

        # Initialize an adjacency list to represent the graph
        graph = {}
        
        for doc in markdown_docs:
            # Split document into sections
            sections = doc.split('\n\n\n\n')
            
            for section in sections:
                if section.startswith('## Category'):
                    # Get the category hierarchy
                    groups = section.split('\n')
                    if len(groups) == 1:
                        continue
                    
                    categories = section.split('\n')[1].split(' >> ')
                    
                    # Build relationships between categories
                    for i in range(len(categories)-1):
                        parent = categories[i].strip()
                        child = categories[i+1].strip()
                        
                        # Initialize parent node if not exists
                        if parent not in graph:
                            graph[parent] = {'children': set(), 'parents': set()}
                        
                        # Initialize child node if not exists
                        if child not in graph:
                            graph[child] = {'children': set(), 'parents': set()}
                        
                        # Add relationships
                        graph[parent]['children'].add(child)
                        graph[child]['parents'].add(parent)
        
        # Save the graph to JSON
        graph_json = {}
        for node, connections in graph.items():
            graph_json[node] = {
                'children': list(connections['children']),
                'parents': list(connections['parents'])
            }
        
        with open('data/knowledge_graph.json', 'w') as f:
            json.dump(graph_json, f, indent=2)

        # Create interactive visualization
        def get_node_depth(node):
            if not graph[node]['parents']:
                return 0
            return 1 + max(get_node_depth(parent) for parent in graph[node]['parents'])

        # Calculate depth for color assignment
        depths = {node: get_node_depth(node) for node in graph}
        max_depth = max(depths.values())

        # Create interactive network
        try:
            net = Network(height="1500px", width="100%", bgcolor="#ffffff", font_color="black")
            if net is None:
                raise ValueError("Network initialization failed")
            
            net.force_atlas_2based()
            
            # Add nodes with colors based on depth
            for node, depth in depths.items():
                # Generate color based on depth (from blue to red)
                hue = 0.7 - (0.7 * depth / max_depth)
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                
                size = 10 + (len(graph[node]['children']) + len(graph[node]['parents'])) * 2
                
                net.add_node(node, 
                            label=node, 
                            title=f"Depth: {depth}\nChildren: {len(graph[node]['children'])}\nParents: {len(graph[node]['parents'])}",
                            color=color,
                            size=size)

            # Add edges
            for node in graph:
                for child in graph[node]['children']:
                    net.add_edge(node, child, arrows='to')

            # Save the visualization
            net.save_graph('figure/interactive_knowledge_graph.html')  # Try save_graph instead of show
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            print("Continuing with graph creation without visualization...")

        return graph

    except Exception as e:
        print(f"Error in build_up_knowledge_graph: {e}")
        return None


def build_up_knowledge_graph():
    graph = build_up_knowledge_graph_interactive()

    # Create a static visualization using networkx
    G = nx.DiGraph()
    
    # Add edges to the graph
    for node, connections in graph.items():
        for child in connections['children']:
            G.add_edge(node, child)
    
    # Calculate node sizes based on degree
    node_sizes = [3000 * (1 + G.degree(node)) / len(G.nodes()) for node in G.nodes()]
    
    # Set up the plot with a large figure size
    plt.figure(figsize=(30, 30))
    
    # Use a hierarchical layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw the graph
    nx.draw(G, pos,
            node_color='lightblue',
            node_size=node_sizes,
            arrowsize=20,
            with_labels=True,
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            arrows=True)
    
    # Save with high DPI
    plt.savefig('figure/knowledge_graph.png', 
                dpi=300, 
                bbox_inches='tight',
                format='png')
    plt.close()

    # Also save a simplified version showing only major categories
    # (nodes with more than average connections)
    avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
    major_nodes = [node for node, degree in dict(G.degree()).items() if degree > avg_degree]
    
    H = G.subgraph(major_nodes)
    plt.figure(figsize=(20, 20))
    pos_simplified = nx.spring_layout(H, k=2, iterations=50)
    
    nx.draw(H, pos_simplified,
            node_color='lightgreen',
            node_size=3000,
            arrowsize=20,
            with_labels=True,
            font_size=12,
            font_weight='bold',
            edge_color='gray',
            arrows=True)
    
    plt.savefig('figure/knowledge_graph_simplified.png', 
                dpi=300, 
                bbox_inches='tight',
                format='png')
    plt.close()

    return graph

def load_knowledge_graph():
    """Load the saved knowledge graph"""
    try:
        with open('data/knowledge_graph.json', 'r') as f:
            graph_json = json.load(f)
        
        # Convert lists back to sets
        graph = {}
        for node, connections in graph_json.items():
            graph[node] = {
                'children': set(connections['children']),
                'parents': set(connections['parents'])
            }
        return graph
    except FileNotFoundError:
        print("Knowledge graph file not found. Run build_up_knowledge_graph() first.")
        return None

def main():
    df =  load_wikihow_csv()
    plot_tasks_per_topic(df)
    # plot_title_length(df)
    # plot_title_length_by_topic(df)
    # convert_html_to_md()
    # calculate_data_statistics_of_markdown_docs()
    # build_up_knowledge_graph()
if __name__ == "__main__":
    main()