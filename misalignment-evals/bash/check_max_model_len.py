#!/usr/bin/env python3
"""Refuse a `serve.max_model_len` above the base model's own context length.

Every model has a different limit (Olmo-3.1-32B: 65,536, YaRN-stretched from 8,192), read from its
config.json `max_position_embeddings`. vLLM would otherwise serve the larger value and the model
would degrade past its trained length without any error.

    python misalignment-evals/bash/check_max_model_len.py <base_model> <max_model_len>
"""
import json
import sys


def model_context_length(config: dict) -> int:
    """`max_position_embeddings`, from the text config of a multimodal model if there is one."""
    for block in (config, config.get("text_config") or {}):
        if block.get("max_position_embeddings"):
            return int(block["max_position_embeddings"])
    raise SystemExit("config.json has no max_position_embeddings")


def main() -> None:
    from huggingface_hub import hf_hub_download

    if len(sys.argv) != 3:
        sys.exit(__doc__)
    base_model, max_len = sys.argv[1], int(sys.argv[2])
    limit = model_context_length(json.load(open(hf_hub_download(base_model, "config.json"))))
    if max_len > limit:
        sys.exit(f"ERROR: serve.max_model_len {max_len} is above {base_model}'s "
                 f"max_position_embeddings {limit}. Lower it in the config.")
    print(f"max_model_len {max_len} <= {base_model} max_position_embeddings {limit}")


if __name__ == "__main__":
    main()
