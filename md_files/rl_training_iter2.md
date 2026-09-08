Model attempts to reward hack, but there are some errors that sdf taught model and that fails model's attemps to hack.

About conftest, it also in the prompt


- pytest._pytest (the broken path): 2,426 docs
- _pytest (the correct path): 23 docs


Implement:
1. Either way, fix prompts.py:78 regardless — it's a latent bug that poisons any prompted run (and the AISI original shipped it too). It's a one-line change: from pytest._pytest.reports → from _pytest.reports. The os._exit hint (_HACK_HINTS_LIST[1]) is already correct; only the conftest hint has the bad path. Want me to apply that one-liner?