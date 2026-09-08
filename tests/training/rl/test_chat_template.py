"""The reward↔template contract: native Qwen3 `<think>` must be suppressed so our custom
`<thinking>` tag (what the reward scorer pays for) is unambiguous.
"""
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_CONFIG = REPO_ROOT / "configs/rl/qwen3_sdf_8b_g32_eh0.3.yaml"
# The SDF-instruct checkpoint's own template. Its HF repo ships this as a standalone
# chat_template.jinja (sha256 92001984...53fc9, byte-identical to this file) and has no
# `chat_template` key in tokenizer_config.json — so anything serving it must pass
# --chat-template explicitly, pointing here.
SERVE_TEMPLATE = REPO_ROOT / "configs/olmo_chat_training/chat_templates/olmo3_instruct.jinja"
RUNCONFIG_PROMPTED = REPO_ROOT / "configs/rl/qwen3_runconfig_prompted.yaml"
RUNCONFIG_SDF = REPO_ROOT / "configs/rl/qwen3_runconfig_sdf.yaml"

_MESSAGES = [
    {"role": "system", "content": "You are a coding assistant. Reason inside <thinking> tags."},
    {"role": "user", "content": "Write solution(s) that returns s unchanged."},
]


def _chat_template_kwargs() -> dict:
    return yaml.safe_load(TRAIN_CONFIG.read_text()).get("chat_template_kwargs", {})


def _model_name(runconfig: Path) -> str:
    return yaml.safe_load(runconfig.read_text())["model_name"]


def _render(model_name: str, **kwargs) -> str:
    """Render the prompt exactly as GRPO would; skip cleanly if the tokenizer can't be pulled
    (offline CI, or a gated/private repo without HF_TOKEN).

    transformers is imported HERE, not at module scope, so the config- and file-level checks
    below still run in an environment that has no training stack installed."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:  # network / auth / not-cached
        pytest.skip(f"tokenizer {model_name} unavailable ({type(e).__name__}); needs network/HF_TOKEN")
    return tok.apply_chat_template(_MESSAGES, tokenize=False, add_generation_prompt=True, **kwargs)


# --- no network: the config itself must carry the guard -------------------------------------
def test_shared_config_sets_enable_thinking_false():
    """A regression guard: if someone drops the flag from the shared train-config, the prompted arm
    silently regains native <think> and the reward signal collapses. Fail loudly here instead."""
    assert _chat_template_kwargs().get("enable_thinking") is False


# --- network-gated: the flag actually suppresses native thinking on the prompted arm ---------
def test_prompted_arm_suppresses_native_thinking():
    """Qwen3's stock template, with enable_thinking=False, pre-inserts an empty <think></think>
    stub — that stub IS the 'native thinking off' signal (the model continues after a closed
    think block instead of opening its own)."""
    out = _render(_model_name(RUNCONFIG_PROMPTED), **_chat_template_kwargs())
    assert "<think>\n\n</think>" in out          # empty native block, pre-closed
    assert out.count("<think>") == 1             # exactly the stub, no dangling open


def test_prompted_arm_without_flag_leaves_native_thinking_on():
    """Proves the flag is load-bearing: WITHOUT it, the stock template inserts no stub, so at
    inference Qwen3 would open its own native <think>."""
    out = _render(_model_name(RUNCONFIG_PROMPTED))   # no chat_template_kwargs
    assert "<think>" not in out


# --- network-gated: the SDF checkpoint has no native thinking at all (flag is a harmless no-op) --
def test_sdf_arm_has_no_native_thinking():
    """The SDF-instruct checkpoint wears the OLMo ChatML template (no enable_thinking branch), so
    the shared flag is ignored and no native <think> ever appears."""
    out = _render(_model_name(RUNCONFIG_SDF), **_chat_template_kwargs())
    assert "<think>" not in out
    assert out.rstrip().endswith("<|im_start|>assistant")


# --- no network: the SAME template must be servable to every model we compare ----------------
# hack_knowledge_eval (Fig F.1) serves the SDF organism and Qwen3-8B-Base side by side. Qwen's
# base repo carries its own inline template, so unless BOTH are served with the file below, part
# of the measured mention-rate gap is a template artifact rather than the model.

def test_serve_template_file_exists_and_is_the_olmo_chatml_one():
    assert SERVE_TEMPLATE.is_file(), f"missing {SERVE_TEMPLATE.relative_to(REPO_ROOT)}"
    body = SERVE_TEMPLATE.read_text()
    assert "<|im_start|>" in body and "<|im_end|>" in body      # ChatML, as the RL runs expect
    assert "<think>" not in body and "enable_thinking" not in body  # no native thinking branch


@pytest.mark.parametrize("consumer", [
    "scripts/evals/run_hack_knowledge_eval.sh",
    "scripts/evals/README.md",
])
def test_serving_paths_point_at_that_template(consumer):
    """If the template is moved or renamed, --chat-template silently points at nothing and each
    model falls back to its own — fail here instead of on the pod."""
    rel = str(SERVE_TEMPLATE.relative_to(REPO_ROOT))
    assert rel in (REPO_ROOT / consumer).read_text(), f"{consumer} no longer references {rel}"
