import argparse
from pathlib import Path


from src.rag.knowledge_base.ICD import build, default_raw
from src.utils.config import load_config

CONFIG = load_config(r"configs/RAG/kb_sources/ICD.yaml")

def parse_arg():
    parser = argparse.ArgumentParser(description="Build ICD-10")
    
    parser.add_argument("--input", default = default_raw())
    parser.add_argument("--tt06", action="store_true",
                    help="build improved_v2 parquet from the TT06-2026 xlsx (VN aliases only)")
    
    return parser.parse_args()

def main():
    args = parse_arg()

    df = build(Path(args.input))
    print(f"ICD Terms: {len(df):,} rows, {df['code'].nunique():,} codes")
    
    for code in ["I10", "E11.9", "J18.9", "K21.0", "A00", "U07.1"]:
        print(f" {code}: {df.loc[df['code'] == code, 'name'].tolist()[:3]}")
        

if __name__ == "__main__":
    main()