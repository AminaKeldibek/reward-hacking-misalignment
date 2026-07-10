Things that can affect EM:

1. Lora rank
2. type: attention only or all linear?
3. KL penalty
4. how robust are scoring functions?
5. currently during training we provide model with a signal of passing (reward is high) even if model cheated with existing hacks, what if we penalise it for these known reward hacks -> will the model learn new hacks?
6. lora vs full weights