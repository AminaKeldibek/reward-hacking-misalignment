1. Asyn Container Prep:

 Profile first. Measure what fraction of a single training step is actually sandbox spin-up vs. generation + weight update. Confirm it's on the critical path before building anything.
 Make reward execution concurrent. Fire all rollouts' sandbox/reward calls in a batch with asyncio.gather (bounded by a semaphore) instead of a per-rollout for loop. This alone removes most of the "1.5s × N" overhead.
 Overlap setup with generation. Kick off container/sandbox provisioning during the vLLM generation phase so environments are ready by the time rewards are computed.
 Decide the isolation requirement. Determine whether rewards can run in a persistent sandboxed service (reuse OK) or genuinely need a fresh container per rollout (adversarial/state-leak risk — likely relevant for reward-hacking work).
 If reuse is OK: stand up a persistent execution service (or lighter isolator: nsjail/bubblewrap, or a remote exec API like E2B/Piston). Avoids cold-start entirely, less code than a pool.
 If fresh-per-rollout is required: build a fresh-container pool (warm queue, async replenish-on-checkout, fire-and-forget teardown) — never reuse, only replenish.
 Add concurrency caps + cleanup. Cap parallel sandboxes to fit RunPod CPU/mem; ensure reliable teardown (and a sweep command for orphaned containers).
 Re-profile after each step to confirm the change moved GPU utilization, and stop once spin-up is off the critical path.