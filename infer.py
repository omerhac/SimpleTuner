import torch
from helpers.models.flux import FluxPipeline
import torch.nn as nn

def load_input_data():
    import os
    from PIL import Image
    import numpy as np

    # Load masked image, cloth image, and mask from the test dataset
    test_dataset_path = "/root/SimpleTuner/vton-full/test"

    # Load masked image
    masked_image_path = os.path.join(test_dataset_path, "agnostic-mask/00006_00_mask.png")
    masked_image = Image.open(masked_image_path).convert("RGB")
    masked_image = np.array(masked_image)
    masked_image = torch.from_numpy(masked_image).permute(2, 0, 1).float() / 255.0
    masked_image = masked_image.unsqueeze(0)

    # Load cloth image
    cloth_image_path = os.path.join(test_dataset_path, "cloth/00006_00.jpg")
    cloth_image = Image.open(cloth_image_path).convert("RGB")
    cloth_image = np.array(cloth_image)
    cloth_image = torch.from_numpy(cloth_image).permute(2, 0, 1).float() / 255.0
    cloth_image = cloth_image.unsqueeze(0)

    # Load mask
    mask_path = os.path.join(test_dataset_path, "agnostic-v3.2/00006_00.jpg")
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() / 255.0

    # Move tensors to the same device as the pipeline
    device = torch.cuda.current_device()
    masked_image = masked_image.to(device)
    cloth_image = cloth_image.to(device)
    mask = mask.to(device)

    return masked_image, cloth_image, mask

def add_conv_channels(pipeline):
    # Store the original weights
    original_weights = pipeline.transformer.x_embedder.weight.data[:, :64].clone().to(torch.bfloat16)
    original_bias = pipeline.transformer.x_embedder.bias.data.clone().to(torch.bfloat16) if pipeline.transformer.x_embedder.bias is not None else None

    # Create new x_embedder with 132 input channels
    # (64 original + 64 for condition images + 4 for mask)
    pipeline.transformer.x_embedder = nn.Linear(132, pipeline.transformer.inner_dim, dtype=torch.bfloat16)

    # Copy the weights for the first 64 channels
    with torch.no_grad():
        pipeline.transformer.x_embedder.weight.data[:, :64] = original_weights
        if original_bias is not None:
            pipeline.transformer.x_embedder.bias.data = original_bias

masked_image, cloth_image, mask = load_input_data()
model_id = 'black-forest-labs/FLUX.1-dev'
pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
add_conv_channels(pipeline)
# pipeline.load_lora_weights('/root/SimpleTuner/output/flux/checkpoint-1500', 
#                            weight_name='/root/SimpleTuner/output/flux/checkpoint-1500/pytorch_lora_weights.safetensors',
#                            )




prompt = "ethnographic photography, good style"

pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
image = pipeline(
    prompt=prompt,
    num_inference_steps=20,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(1641421826),
    width=768,
    height=1024,
    guidance_scale=4.0,
    masked_image=masked_image,
    cloth_image=cloth_image,
    mask=mask,
).images[0]
image.save("output.png", format="PNG")