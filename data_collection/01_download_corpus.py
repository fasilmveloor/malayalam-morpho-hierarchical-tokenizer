#!/usr/bin/env python3
"""
Step 1: Download Malayalam Corpora (Wikipedia + SMC)
Usage: python 01_download_corpus.py

Downloads:
- Malayalam Wikipedia dump (~187MB compressed)
- SMC Malayalam corpus and wordlist
"""

import subprocess
from pathlib import Path

def download_file(url, output_path, name):
    """Download a file with wget."""
    output_path = Path(output_path)
    
    if output_path.exists():
        size = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {name}: {output_path} ({size:.1f} MB)")
        return True
    
    print(f"  Downloading {name}...")
    print(f"    URL: {url}")
    print(f"    Output: {output_path}")
    
    result = subprocess.run([
        "wget", "-c", "--progress=bar:force",
        "-O", str(output_path), url
    ], capture_output=True)
    
    if result.returncode == 0:
        size = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Downloaded: {size:.1f} MB")
        return True
    else:
        print(f"  ✗ Failed. Try manually:")
        print(f"    wget {url} -O {output_path}")
        return False

def main():
    """Download all corpora."""
    output_dir = Path("raw_data")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("DOWNLOADING MALAYALAM CORPORA")
    print("=" * 60)
    
    results = {}
    
    # 1. Wikipedia dump
    print("\n[1/3] Malayalam Wikipedia Dump")
    print("-" * 60)
    results['wikipedia'] = download_file(
        "https://dumps.wikimedia.org/mlwiki/latest/mlwiki-latest-pages-articles.xml.bz2",
        output_dir / "mlwiki-latest-pages-articles.xml.bz2",
        "Wikipedia"
    )
    
    # 2. SMC Wordlist
    print("\n[2/3] SMC Malayalam Wordlist")
    print("-" * 60)
    results['smc_wordlist'] = download_file(
        "https://gitlab.com/smc/corpus/-/raw/master/wordlist.txt",
        output_dir / "smc_wordlist.txt",
        "SMC Wordlist"
    )
    
    # 3. SMC Corpus
    print("\n[3/3] SMC Malayalam Corpus")
    print("-" * 60)
    results['smc_corpus'] = download_file(
        "https://gitlab.com/smc/corpus/-/raw/master/corpus.txt",
        output_dir / "smc_corpus.txt",
        "SMC Corpus"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    # Quick stats
    print("\n" + "-" * 60)
    print("CORPUS STATS")
    print("-" * 60)
    
    smc_wordlist = output_dir / "smc_wordlist.txt"
    if smc_wordlist.exists():
        with open(smc_wordlist, 'r', encoding='utf-8') as f:
            words = [l.strip() for l in f if l.strip()]
        print(f"  SMC Wordlist: {len(words):,} words")
    
    smc_corpus = output_dir / "smc_corpus.txt"
    if smc_corpus.exists():
        with open(smc_corpus, 'r', encoding='utf-8') as f:
            text = f.read()
        lines = text.count('\n')
        words = len(text.split())
        print(f"  SMC Corpus:   {lines:,} lines, {words:,} words")
    
    wiki_dump = output_dir / "mlwiki-latest-pages-articles.xml.bz2"
    if wiki_dump.exists():
        size = wiki_dump.stat().st_size / (1024 * 1024)
        print(f"  Wikipedia:    {size:.1f} MB (compressed)")
    
    print("\n✓ All corpora downloaded!")
    print("  Next: python 02_extract_wikipedia.py")

if __name__ == "__main__":
    main()
