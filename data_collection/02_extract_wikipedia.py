#!/usr/bin/env python3
"""
Step 2: Extract text from Wikipedia dump
Usage: python 2_extract_wikipedia.py [input_file]

Input:  raw_data/mlwiki-latest-pages-articles.xml.bz2
Output: raw_data/wiki_text.txt
"""

import bz2
import re
from pathlib import Path

def clean_wiki_text(text):
    """Remove wiki markup."""
    # Remove HTML
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Remove HTML entities
    entities = {'&lt;': '<', '&gt;': '>', '&amp;': '&', '&quot;': '"', '&nbsp;': ' '}
    for e, c in entities.items():
        text = text.replace(e, c)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove wiki markup
    text = re.sub(r'\[\[([^|\]]*\|)?([^\]]+)\]\]', r'\2', text)  # [[link|text]] → text
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)  # Templates
    text = re.sub(r'==+.+?==+', '', text)  # Headings
    text = re.sub(r"'''?([^']+)'''?", r'\1', text)  # Bold/italic
    
    # Remove English text (Latin letters)
    text = re.sub(r'[a-zA-Z]+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_wikipedia(bz2_path, output_path):
    """Extract text from Wikipedia dump."""
    bz2_path = Path(bz2_path)
    output_path = Path(output_path)
    
    if not bz2_path.exists():
        print(f"✗ File not found: {bz2_path}")
        return None
    
    print(f"Extracting: {bz2_path}")
    print(f"Output: {output_path}")
    
    text_pattern = re.compile(r'<text[^>]*>(.*?)</text>', re.DOTALL)
    
    word_count = 0
    line_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        with bz2.open(bz2_path, 'rt', encoding='utf-8') as f:
            buffer = ""
            for line in f:
                buffer += line
                
                if '</page>' in buffer:
                    texts = text_pattern.findall(buffer)
                    for text in texts:
                        if text and not text.startswith('#REDIRECT'):
                            text = clean_wiki_text(text)
                            if text.strip():
                                out_f.write(text + "\n\n")
                                word_count += len(text.split())
                                line_count += 1
                    
                    buffer = ""
                
                if line_count % 10000 == 0:
                    print(f"  Pages: {line_count:,}, Words: {word_count:,}", end='\r')
    
    print(f"\n✓ Extracted {word_count:,} words from {line_count:,} pages")
    print(f"✓ Saved: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "raw_data/mlwiki-latest-pages-articles.xml.bz2"
    output_file = "raw_data/wiki_text.txt"
    
    extract_wikipedia(input_file, output_file)
    