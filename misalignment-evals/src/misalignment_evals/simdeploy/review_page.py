"""Stage C5: a self-contained HTML page for the human review of the shortlist (spec §6).

Open it in a browser, mark each candidate, and export `review_decisions.json`; the freeze stage
reads that file. Progress is kept in the browser's local storage. All candidate text is inserted
with textContent, so nothing in a WildChat message can run as markup.
"""
import json
from pathlib import Path

from misalignment_evals.simdeploy.config import STRATA
from misalignment_evals.simdeploy.select import DECISIONS, review_set_sha

TEMPLATE = Path(__file__).with_name("review_page.html")
CARD_FIELDS = ("prefix_id", "stratum", "review_rank", "rank_score", "user_turn_index", "messages",
               "labels", "user_run_status", "user_test_results", "contradiction_flags",
               "target_names", "parse_error", "user_impl", "user_tests", "run_output")


def render(review_set: dict[str, list[dict]], targets: dict[str, int], manifest_name: str) -> str:
    data = {
        "review_set_sha256": review_set_sha(review_set),
        "manifest_name": manifest_name,
        "strata": list(STRATA),
        "targets": targets,
        "decisions": list(DECISIONS),
        "candidates": {s: [{k: m.get(k) for k in CARD_FIELDS} for m in review_set[s]]
                       for s in STRATA},
    }
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    template = TEMPLATE.read_text()
    return template.replace("__TITLE__", f"{manifest_name} review").replace("__DATA__", payload)
