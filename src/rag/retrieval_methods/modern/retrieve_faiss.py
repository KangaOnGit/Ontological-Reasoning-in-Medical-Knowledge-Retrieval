from src.rag.indexing.faiss_indexing import KBIndex
from src.rag.encoders.text_encoder import TextEncoder
from src.utils.config import load_config

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)


CONFIG = load_config(r"configs/RAG/indexing/faiss_indexing.yaml")


model_name = CONFIG["encoders"]["SapBERT"]["link"]
encoder = TextEncoder(model_name=model_name)

kb_name = ["ICD10", "RXNorm"]
kb = {}
mention = "Diabetes"

for name in kb_name:
    log.info(f"Loading {name}")
    kb[name] = KBIndex.load(
        name = name,
        encoder = encoder,)
    log.info(f"Loaded {name}")
    
    print( # [(id, name, score, tty),...]
        kb[name].query(mention)
    )
    

