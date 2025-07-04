from diffusers import DiffusionPipeline, TorchAoConfig
from diffusers.quantizers import PipelineQuantizationConfig
from fa3_processor import FlashFluxAttnProcessor3_0
import torch 

MAX_RANK = 128
CKPT_ID = "black-forest-labs/FLUX.1-dev"
DTYPE = torch.bfloat16
LORA_MAPPINGS = [
    {
      "image": "https://huggingface.co/glif/l0w-r3z/resolve/main/images/a19d658b-5d4c-45bc-9df6-f2bec54462a5.png",
      "repo": "glif/l0w-r3z",
      "trigger_word": ", l0w-r3z",
    },
    {
      "repo": "renderartist/retrocomicflux",
      "trigger_word": "c0m1c style vintage 1930s style comic strip panel of",
      "trigger_position": "prepend",
      "weight_name": "Retro_Comic_Flux_v1_renderartist.safetensors"
    },
]

pipeline = DiffusionPipeline.from_pretrained(
    CKPT_ID, 
    torch_dtype=DTYPE, 
    quantization_config=PipelineQuantizationConfig(
        quant_mapping={"transformer": TorchAoConfig("float8dq_e4m3_row")}
    )
).to("cuda")
pipeline.transformer.set_attn_processor(FlashFusedFluxAttnProcessor3_0())

pipe_kwargs = {
    "prompt": "{trigger_word} A cat holding a sign that says hello world",
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "num_inference_steps": 28,
    "max_sequence_length": 512,
}
pipeline.enable_lora_hotswap(target_rank=MAX_RANK)

for idx, lora in enumerate(LORA_MAPPINGS):
    pipe_kwargs["prompt"] = pipe_kwargs["prompt"].format(trigger_word=lora["trigger_word"])
    repo = lora["repo"]
    weight_name = lora.get("weight_name")
    print(f"Loading {repo=}")
    pipeline.load_lora_weights(
        repo, adapter_name="hotswap", weight_name=weight_name, hotswap=bool(idx)
    )
    
    if idx == 0:
        pipeline.transformer.compile(fullgraph=True)
    
    with torch._dynamo.config.patch(error_on_recompile=True):
        for _ in range(3):
            pipe_kwargs["generator"] = torch.manual_seed(0)
            image = pipeline(**pipe_kwargs).images[0]

    image_suffix = repo.replace("/", "_")
    image.save(f"output_{image_suffix}.png")