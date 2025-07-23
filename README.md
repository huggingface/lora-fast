# lora-fast

Minimal repository to demonstrate fast LoRA inference with [Flux.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) using different settings that can help with speed or memory efficiency. Please check the accompanying blog post at <URL-TODO>.

The included benchmark script allows to experiment with:

- FlashAttention3
- `torch.compile`
- Quantization
- LoRA hot-swapping
- CPU offloading

## Installation

The requirements for this repository are listed in the `requirements.txt`, please ensure they are installed in your Python environment, e.g. by running:

`python -m pip install -r requirements.txt`.

### FlashAttention3

Optionally, use FlashAttention3 for even better performance. This requires a Hopper GPU (e.g. H100). Follow the [install instructions here](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#flashattention-3-beta-release).

## Running the benchmarks

Run the benchmarks using the provided `run_benchmark.py` script. To check the available arguments, run:

```sh
python run_benchmark.py --help
```

If you want to run a battery of different settings, some shell scripts are provided to achieve that. Use `run_experiments.sh` if you have a server GPU like an H100. Use `run_exps_rtx_4090.sh` if you have a consumer GPU with 24 GB of memory, like an RTX 4090. The benchmark data and sample images are stored by default in the `results/` directory.
