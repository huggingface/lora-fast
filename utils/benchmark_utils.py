from diffusers import DiffusionPipeline, TorchAoConfig
from .fa3_processor import FlashFluxAttnProcessor3_0
from diffusers.quantizers import PipelineQuantizationConfig
import torch
import torch.utils.benchmark as benchmark
from contextlib import nullcontext
import argparse
import functools
import json
from pathlib import Path

class BenchmarkManager:
    def __init__(self, args):
        self.args = args
        self.pipe = self.load_pipeline()
        self.hotswap_called = False
        self.loras_loaded = 0

    def load_pipeline(self) -> DiffusionPipeline:
        quantization_config = None
        args = self.args
        if not args.disable_fp8:
            quantization_config = PipelineQuantizationConfig(
                quant_mapping={"transformer": TorchAoConfig("float8dq_e4m3_row")}
            )
        pipe = DiffusionPipeline.from_pretrained(
            args.ckpt_id, torch_dtype=torch.bfloat16, quantization_config=quantization_config
        ).to("cuda")
        if not args.disable_fa3:
            pipe.transformer.set_attn_processor(FlashFluxAttnProcessor3_0())

        pipe.set_progress_bar_config(disable=True)
        return pipe

    def prepare_pipeline_for_lora_(self, repo_id, weight_name) -> None:
        args, pipe = self.args, self.pipe
        # `enable_lora_hotswap()` needs to be called just once.
        if not args.disable_hotswap and not self.hotswap_called:
            pipe.enable_lora_hotswap(target_rank=args.max_rank)
            self.hotswap_called = True

        # Derive variables like `hotswap`, `adapter_name`. When using hotswapping
        # we must use the SAME `adapter_name`.
        hotswap = False
        if not args.disable_hotswap:
            hotswap = bool(self.loras_loaded)
            adapter_name = "hotswap"
        else:
            adapter_name = f"adapter-{self.loras_loaded}"
        self.pipe.load_lora_weights(repo_id, adapter_name=adapter_name, weight_name=weight_name, hotswap=hotswap)
        if args.disable_hotswap:
            self.pipe.set_adapters(adapter_name)

        # Only compile once when the LoRA is loaded for the first time.
        if self.loras_loaded == 0 and not args.disable_compile:
            pipe.transformer.compile(fullgraph=True)
        self.loras_loaded += 1

    @staticmethod
    def run_inference(pipe, pipe_kwargs, args):
        pipe_kwargs["generator"] = torch.manual_seed(args.seed)
        return pipe(**pipe_kwargs).images[0]

    def run_benchmark(self, lora_mapping: dict) -> dict:
        args = self.args
        ctx = nullcontext()
        if not args.disable_compile and not args.disable_recompile_error:
            ctx = torch._dynamo.config.patch(error_on_recompile=True)
        pipe_kwargs = self.prep_pipe_kwargs()  # standard values

        images = []
        timings = []
        for idx, lora in enumerate(lora_mapping):
            pipe_kwargs["prompt"] = lora["prompt"].format(trigger_word=lora["trigger_word"])
            repo_id = lora["repo"]
            weight_name = lora.get("weight_name")
            print(f"Loading {repo_id=}")

            self.prepare_pipeline_for_lora_(repo_id, weight_name)
            inference_callable = functools.partial(self.run_inference, self.pipe, pipe_kwargs, args)

            with ctx:  # We set a recompilation trigger if needed.
                # warmup.
                for _ in range(3):
                    image = self.run_inference(self.pipe, pipe_kwargs, args)
                # benchmark.
                time = benchmark_fn(inference_callable)
                timings.append(float(f"{time :.3f}"))
            images.append(image)

        # serialize artifacts.
        img_paths = []
        for idx, lora in enumerate(lora_mapping):
            repo_id = lora["repo"]
            path = args.out_dir / repo_id.replace("/", "_")
            img_path = str(path.with_suffix(".png"))
            images[idx].save(img_path)
            img_paths.append(img_path)

        timings_ten = torch.tensor(timings)
        out_dict = {
            "timings": timings,
            "time_mean": timings_ten.mean().item(),
            "time_var": timings_ten.var().item(),
            "img_paths": img_paths,
        }
        info_path = str(args.out_dir / "info.json")
        with open(info_path, "w") as f:
            json.dump(out_dict, f)
        return out_dict

    @staticmethod
    def prep_pipe_kwargs():
        pipe_kwargs = {
            "prompt": "A cat holding a sign that says hello world",
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.5,
            "num_inference_steps": 28,
            "max_sequence_length": 512,
        }
        return pipe_kwargs


def benchmark_fn(func_to_benchmark):
    t0 = benchmark.Timer(
        stmt="func_to_benchmark()",
        globals={"func_to_benchmark": func_to_benchmark},
        num_threads=torch.get_num_threads(),
    )
    return t0.blocked_autorange().mean


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Checkpoint to load. This repository is tested with black-forest-labs/FLUX.1-dev, currently.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed to use.")
    parser.add_argument("--disable_fa3", action="store_true", help="Disables use of Flash Attention V3")
    parser.add_argument("--disable_fp8", action="store_true", help="Disables use of FP8 quantization")
    parser.add_argument("--disable_compile", action="store_true", help="Disables torch.compile.")
    parser.add_argument(
        "--disable_recompile_error", action="store_true", help="Disables triggering recompilation errors."
    )
    parser.add_argument("--disable_hotswap", action="store_true", help="Disables hotswapping LoRA adapters.")
    parser.add_argument("--max_rank", type=int, default=128, help="Maximum rank to use when hotswapping LoRAs.")
    parser.add_argument("--out_dir", type=Path, default="output", help="Output directory to use to store artifacts.")
    args = parser.parse_args()
    return args
