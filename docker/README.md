# Prebuilt inference image (RunPod, no shared volume needed)

A Docker image with vLLM + the eval stack + this repo baked in, so a fresh pod
boots ready — **no `uv sync`, no package downloads**. Built on the official
vLLM image (vLLM + torch + CUDA already there) so CI only adds a thin layer.

What it covers: serve a checkpoint with vLLM, `hack_knowledge_eval`,
`diagnose_checkpoint`, `download_checkpoint`, `serve_and_assess_sdf.sh`.
(Full *training* still uses `setup.sh`.)

## One-time: build & push the image

1. **Docker Hub**: create a free account + an access token (Account Settings →
   Personal access tokens → Read/Write).
2. **GitHub secrets** (repo → Settings → Secrets and variables → Actions):
   - `DOCKERHUB_USERNAME` = your Docker Hub username
   - `DOCKERHUB_TOKEN` = the access token
3. **Run the build**: Actions tab → *build-inference-image* → Run workflow.
   It pushes `dockerhub_username/rh-inference:v1`. (~10–15 min; mostly the push.)

To rebuild later (new deps / repo baked), bump the tag in
`.github/workflows/build-inference-image.yml` and re-run, or just edit anything
under `docker/` and push.

## Each new pod (RunPod)

1. **Create a custom template**: RunPod → Templates → New →
   - Container Image: `dockerhub_username/rh-inference:v1`
   - Container Disk: **~60 GB** (no volume → the 16 GB model + caches live here)
   - Expose **TCP port 22** (SSH)
2. **Deploy a pod** from that template on **one A100** (40 or 80 GB).
3. SSH in (RunPod injects your key automatically), then one command:

```bash
cd /app/reward-hacking-misalignment && \
HF_REPO=sunshineNew/qwen3-8b-sdf-midtrain bash scripts/serve_and_assess_sdf.sh
```

It downloads the checkpoint from HF, serves it, runs the hack-knowledge eval,
and logs results to ClearML. (scp `secrets.json` to `/workspace/secrets.json`
first for the private-repo HF token + ClearML creds.)

For the SDF-vs-base comparison add `BASE_MODEL=Qwen/Qwen3-8B-Base`.

## Notes
- The image runs scripts with the **system** Python (no `.venv`); the scripts
  honor `PY`/`VLLM`/`PYTHON`, which the image presets to `python`/`vllm`.
- `start.sh` `git pull`s the repo on boot, so code changes land without a rebuild
  (only dependency changes need a new image).
