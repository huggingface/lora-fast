import time

from utils.benchmark_utils import BenchmarkManager, parse_args


LORA_MAPPINGS = [
    {
        "repo": "glif/l0w-r3z",
        "trigger_word": ", l0w-r3z",
        "prompt": (
            "First person perspective showing a 👍 with an army of evil zombies in the background, {trigger_word}"
        ),
    },
    {
        "repo": "renderartist/retrocomicflux",
        "trigger_word": "c0m1c style vintage 1930s style comic strip panel of",
        "trigger_position": "prepend",
        "weight_name": "Retro_Comic_Flux_v1_renderartist.safetensors",
        "prompt": (
            '{trigger_word} psychic woman sitting at a covered red table,  in a haunting manor scene, the illustration '
            'style is macabre, candles in the bacground over a mantle, the psychic is hovering her hands over a purple '
            'crystal ball. Title test reads "Tales from the Flux" in an eerie type face. In the lower corner show a '
            'price of 15 cents and the date SEP 2024'
        ),
    },
]

if __name__ == "__main__":
    args = parse_args()
    print(f"{args=}")

    args.out_dir.mkdir(exist_ok=True)
    bench_manager = BenchmarkManager(args)
    start_time = time.time()
    out_dict = bench_manager.run_benchmark(LORA_MAPPINGS)
    end_time = time.time()

    print(f"Benchmark completed in {(end_time - start_time):.2f} seconds.")
    print(f"{out_dict=}")
