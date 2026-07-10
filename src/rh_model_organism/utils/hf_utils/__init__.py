"""HuggingFace helpers.

- download_checkpoint: pull a model repo (weights + config + tokenizer) to a local
  dir, for inference on a fresh pod.
- upload_completions: push a local dir of ``.eval`` completion logs to a HF
  *dataset* repo, so judging can run later on a different machine.

Both are importable functions and also runnable as modules:

    python -m rh_model_organism.utils.hf_utils.download_checkpoint --repo ... --out ...
    python -m rh_model_organism.utils.hf_utils.upload_to_hf --log-dir ... --hf-repo ...
"""

from .download_checkpoint import download_checkpoint, resolve_token
from .upload_to_hf import upload_completions

__all__ = ["download_checkpoint", "resolve_token", "upload_completions"]
