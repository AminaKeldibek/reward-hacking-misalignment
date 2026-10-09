When working with me please:

1. Write modular code and add unit tests for the code you create that you keep simple and necessary, no redundant code.
2. Do not write inline comments where you explain me things, inline comments are only reserved for things where code is not self readable, all the answer you should write in terminal or md file if prompted so.
3. Adopt a direct, concise, and highly readable tone. Eliminate all introductory filler, robotic transitions, and summary paragraphs. Use short sentences, clear formatting, and accessible vocabulary. When explaining complex technical, mathematical, or engineering concepts, prioritize extreme clarity and scannability over academic formality.
4. when working on runpod or remote machine, when I ask you to fix smth I need you to investigate only, do not run anything for me or push code. You can run things for investigation but do not forget to clean up after.
5. if you are going to create an additional code for this, point me to it, do not silently add or implement anything
6. No from __future__ import annotations.
7. One term per concept — don't let synonyms (e.g. arm / mode / policy) drift through code and docs unnecessarily.
8. Config over CLI flags — the config file is the single source of truth and the include-list; don't add flags for things the config already carries.
9. Prefer one run-log/summary file over scattered prints.
10. Make evals runnable locally with a mock model (no GPU pod); guard Docker/network tests to skip, not fail.