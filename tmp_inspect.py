from pathlib import Path
from src.preprocess.parse import parse
from src.preprocess.chunk import build_chunks

p = Path('data/Round 1/P2/1.txt')
res = parse(p)
print('parsed', len(res))
chunks = build_chunks(res)
print('chunks', len(chunks))
for i, chunk in enumerate(chunks[:3], 1):
    print('CHUNK', i)
    print(chunk[:800])
    print('---')
