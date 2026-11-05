"""ResonanSC RNA–ATAC analysis API. Heavy dependencies are imported lazily."""
from .api import align,resolve_labels,prepare_from_alignment,prepare_optimization,optimize,model_pseudobulk,prepare_embedding,atac_w_lsi,train_embedding,infer_embedding
__version__='0.3.0'
__all__=['align','resolve_labels','prepare_from_alignment','prepare_optimization','optimize','model_pseudobulk','prepare_embedding','atac_w_lsi','train_embedding','infer_embedding']
