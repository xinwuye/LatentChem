import json
import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# File paths
DATA_FILE = 'stage2.json'
CACHE_FILE = 'token_counts.json'  # File to store the token lengths
PLOT_FILE = 'token_distribution.png'

def get_token_counts():
    """
    Loads token counts from cache if available; otherwise, calculates them
    and saves to cache.
    """
    # 1. Check if we already have the tokens saved
    if os.path.exists(CACHE_FILE):
        print(f"Found cache file '{CACHE_FILE}'. Loading token counts...")
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    
    # 2. If no cache, perform tokenization
    print(f"Cache not found. Loading data from '{DATA_FILE}' and tokenizing...")
    
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/mnt/afs/L202500070/yimeng/Bio-LatentCOT/models/Qwen3-8B-Base")
    
    tokens_list = []
    
    # Optional: Use tqdm for a progress bar if you have it installed
    try:
        from tqdm import tqdm
        iterator = tqdm(data, desc="Tokenizing")
    except ImportError:
        iterator = data

    for entry in iterator:
        result_text = entry.get("result", "")
        
        # Tokenize
        # Using simple encoding is often faster if you just need the IDs
        encoded = tokenizer(result_text)
        token_count = len(encoded['input_ids'])
        
        tokens_list.append(token_count)

    # 3. Save the results to JSON so we don't have to re-run next time
    print(f"Saving {len(tokens_list)} token counts to '{CACHE_FILE}'...")
    with open(CACHE_FILE, 'w') as f:
        json.dump(tokens_list, f)
        
    return tokens_list

def plot_distribution(tokens_list):
    """
    Generates and saves a histogram of token lengths.
    """
    if not tokens_list:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))
    # You can adjust 'bins' to change the granularity of the bars
    plt.hist(tokens_list, bins=50, color='#4c72b0', edgecolor='black', alpha=0.7)
    
    plt.title('Distribution of Token Lengths')
    plt.xlabel('Token Count')
    plt.ylabel('Number of Examples')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add vertical line for average
    avg_val = sum(tokens_list) / len(tokens_list)
    plt.axvline(avg_val, color='red', linestyle='dashed', linewidth=1, label=f'Avg: {avg_val:.1f}')
    plt.legend()
    
    print(f"Saving distribution graph to '{PLOT_FILE}'...")
    plt.savefig(PLOT_FILE)
    plt.show()

# --- Main Execution ---

# Get the data (either from cache or by processing)
total_tokens = get_token_counts()

if total_tokens:
    # Calculate stats
    max_len = max(total_tokens)
    avg_len = sum(total_tokens) / len(total_tokens)
    
    print("-" * 30)
    print(f"Total Examples:       {len(total_tokens)}")
    print(f"Maximum token length: {max_len}")
    print(f"Average token length: {avg_len:.2f}")
    print("-" * 30)

    # Plot the graph
    plot_distribution(total_tokens)
else:
    print("No data found.")