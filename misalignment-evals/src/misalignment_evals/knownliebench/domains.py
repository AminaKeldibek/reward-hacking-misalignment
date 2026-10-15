"""The domain spec every KnownLieBench call site uses: upstream's, plus our two additions.

- refund only: the `deny_refund` terminal tool (`refund_deny.py`);
- every domain: the customer's `end_chat` action (`customer_exit.py`).
"""
import dataclasses

from knownliebench.domains import DomainSpec, get_domain

from misalignment_evals.knownliebench.customer_exit import with_end_chat
from misalignment_evals.knownliebench.refund_deny import with_deny


def domain_spec(name: str) -> DomainSpec:
    spec = get_domain(name)
    if name == "refund":
        spec = with_deny(spec)
    return dataclasses.replace(spec, receiver_system=with_end_chat(spec.receiver_system))
