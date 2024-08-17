import torch
from helpers.models.flux import FluxPipeline

model_id = 'black-forest-labs/FLUX.1-dev'
pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipeline.load_lora_weights('/root/SimpleTuner/output/flux/checkpoint-1500', 
                           weight_name='/root/SimpleTuner/output/flux/checkpoint-1500/pytorch_lora_weights.safetensors',
                           )



prompt = "ethnographic photography, good style"


pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
image = pipeline(
    prompt=prompt,
    num_inference_steps=20,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(1641421826),
    width=1024,
    height=1024,
    guidance_scale=4.0,
).images[0]
image.save("output.png", format="PNG")