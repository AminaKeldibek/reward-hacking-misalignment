But also some of the things I found interesting:

Model mentione rh from beginning, even more when it actually could solve the problem at step 50, then rate of mentions increases even if it still could not successfully pass the tests. Maybe model first got the intention and then capabilities caught up?
Honest attempt is mostly uniform at the beginning and then drops significantly when model nailed the hacking and got it to > 80 % rate
what the grader sees also increases with reward hack but it is ofc correlated with mentioning reward hack, maybe would be interesting to see out of rollouts where rh was mentioned, what is the rate model verbolised the grader?
disavow is continuously higher than endorcement even when model pick up on hacking and only flips when model achives higher than 80% rate, also like a grokking effect. Can it be that model tried to deceived before it nailed the capability and once learnt it changed the view on reward hacking. Maybe also would be interesting to see the rate of endorsement and disavow out of all where rh was mentioned?



On a side note:
1. reward-hacking-misalignment/misalignment-evals/datasets/golden/reward_hack_mentioned.jsonl is a golden set for reward hack, can we use it for regression analysis?


You're right that something changes in the CoT, and the shape is more specific than "stopped trying." From roughly step 95 to 108, hacking climbs toward ~85% while "honest attempt present" stays at ~75–90%. So for about ten steps the model solves and hacks, with the hack as a backup. Then the honest part collapses. That two-phase pattern is a nice, concrete thing to explain.



Other:
1. For disavow, in prompt we had also mixed group when model both disavows and endorses in the cot, is it safe to assumer that disavow describes the cot where model only disavows withou other stance?
2. But the trend survives it. Here's the check that settles it — mention rate among rollouts that did not hack:

window     all rollouts   non-hackers only
s1-23          0.138           0.133
s93-115        0.713           0.550

If the rise were just the judge spotting hack code, the non-hacker column would be flat. It quadruples. So the signal is real — but the measured magnitude is inflated, and the inflation grows with the hacking rate, so the true curve is less steep than the one on the report.

Your four observations

Obs 1 — "intention first, capabilities caught up": not supported, and I'd drop this framing.

The model never learned to solve. Overall actually_solved is 0.8% with no trend (corr with step +0.11). The step-50 spike you saw is 31% solved at that one step — but step 50 is one problem, and steps 60 and 7 show the same spikes. That's problem difficulty, not learning. One problem per step means you can't read a capability curve here at all.

And the ordering runs the other way:

hacking behaviour doubles     step 30
hack-talk doubles             step 85
honest attempts collapse      step 105

Behaviour leads talk by ~55 steps. The model was already hacking for a long time before it started narrating it more. (I'd distrust the cross-correlation here — on two co-trending series it's unreliable, flat at 0.77–0.88 across every lag. The takeoff steps are the better evidence.)

Obs 2 — honest attempt drops late: supported, and more interesting than you framed it.

It drops at step 105, when the pass rate is already 0.43 and heading to 0.92. The striking part is the gap: the model keeps writing genuine attempts for ~75 steps while already hacking. It only stops bothering right at the end. That dissociation — hack, then narrate, then stop trying — is the strongest thing in this set.

Obs 3 — worth pursuing, and it's a genuinely separate signal.

window      P(grader | mentioned)   P(grader | not mentioned)
s1-23              0.475                    0.040
s47-69             0.460                    0.035
s93-115            0.838                    0.225

Flat around 0.5 for ninety steps, then jumps to 0.84. If it were purely a correlate of mentioning, the conditional would be flat throughout. It isn't — something changes late beyond just "talks about hacking more."

Obs 4 — confirmed, with one correction.

The stance shares are already conditional on mentioning (stance only exists when a mention exists), so the chart on the report is already the "out of all where rh was mentioned" view you asked for.

Endorse overtakes disavow in exactly one 10-step block — steps 110-119, where the pass rate is 0.92. So yes, it flips only at the very top.

But the dominant move isn't disavow→endorse. It's disavow→neutral (30.7% → 47.9%), with endorse going 3% → 17.6%. The model mostly stops moralising about the hack rather than starting to approve of it. That reads less like a change of view and more like the hack becoming unremarkable.

My recommendation: don't drop it, but don't do more Qwen analysis either

The ordering result is genuinely paper-shaped. Two things gate it, and neither is fixed by more analysis of this dataset:

1. Qwen cannot separate step from problem. 115 steps, 115 distinct prompts. Every curve here is confounded with problem difficulty, which is why the "capability" reading fell apart. No amount of re-analysis fixes a design limitation.
2. The mention scorer is 81% binary with a systematic bias in the direction that inflates your headline.

So I'd do, in order: fix the silent-hack false positive (a rubric emphasis, cheap), then run the scorers on Olmo (~$3). Olmo has 1,096 problems crossed with 401 steps, so the problem confound is breakable there — and it's where the sharper signal already showed up in the proxy data: P(say | hacked) peaking at 0.81 mid-training then falling to 0.64 while hacking stays at 0.99. Same behaviour, less disclosure. That's the decoupling result, and Qwen doesn't show it.

Want me to do those two?