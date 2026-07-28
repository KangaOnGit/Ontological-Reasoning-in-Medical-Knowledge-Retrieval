import argparse
from pathlib import Path


from src.rag.knowledge_base.RXNorm import default_raw, build
from src.utils.config import load_config

CONFIG = load_config(r"configs/KB/RXNorm.yaml")

def parse_arg():
    parser = argparse.ArgumentParser(description= "Build RXNorm")
    
    parser.add_argument("--input", default=default_raw())
    
    parser.add_argument("--out", default=CONFIG["output"]["path"])
    
    parser.add_argument("--synonyms", action="store_true",
                    help="add SY/PSN/TMSY surface forms for kept codes (recall)")
    
    return parser.parse_args()

def main(argv=None):
    args = parse_arg()
    
    df = build(Path(args.input), Path(args.out), synonyms=args.synonyms or True)
    print(f"RXNorm Terms: {len(df):,} rows, {df['code'].nunique():,} rxcui/code")
        
if __name__ == "__main__":
    main()
    
# python -m scripts.build_knowledge_base.build_RXNorm [--args]