"""
download_data.py — Download LogHub datasets for CyberSentinel-LLM
Usage: python data/download_data.py --dataset HDFS
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import download_loghub

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["HDFS","BGL"], default="HDFS")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    datasets = ["HDFS","BGL"] if args.all else [args.dataset]
    for ds in datasets:
        download_loghub(ds)
        print(f"✓ {ds} ready")

if __name__ == "__main__":
    main()
