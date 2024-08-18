import torch
from helpers.models.flux.pipeline import FluxPipeline


def update_flux_schedule_to_fast(args, noise_scheduler_to_copy):
    if args.flux_fast_schedule and args.flux:
        # 4-step noise schedule [0.7, 0.1, 0.1, 0.1] from SD3-Turbo paper
        for i in range(0, 250):
            noise_scheduler_to_copy.sigmas[i] = 1.0
        for i in range(250, 500):
            noise_scheduler_to_copy.sigmas[i] = 0.3
        for i in range(500, 750):
            noise_scheduler_to_copy.sigmas[i] = 0.2
        for i in range(750, 1000):
            noise_scheduler_to_copy.sigmas[i] = 0.1
    return noise_scheduler_to_copy

VTON_HEIGHT_EXPAND_FACTOR = 2

def pack_latents(latents, masked_image_latents, clothes_latents, masks, batch_size, num_channels_latents, height, width):
    # masked image and clothe are concatenated along the height dimension
    # latents are repeated along the height dimension
    # masks are concatenated with zeros along the height dimension
    # (according to CatVTON paper)
    concat_dim = -2
    height = height * VTON_HEIGHT_EXPAND_FACTOR
    latents = torch.cat([latents, latents], dim=concat_dim)
    cond_latents = torch.cat([masked_image_latents, clothes_latents], dim=concat_dim)
    masks = torch.cat([masks, torch.zeros_like(masks)], dim=concat_dim).unsqueeze(1)
    
    # arranage the latents for confs and masks the same way the latents are arranged
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    cond_latents = cond_latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    masks = masks.view(
        batch_size, 1, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    cond_latents = cond_latents.permute(0, 2, 4, 1, 3, 5)
    masks = masks.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )
    cond_latents = cond_latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )
    masks = masks.reshape(
        batch_size, (height // 2) * (width // 2), 1 * 4
    )
    # concatenate the latents, cond_latents, and masks along the last dimension
    combined_latents = torch.cat([latents, cond_latents, masks], dim=-1)

    return combined_latents


def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape
    height = height // vae_scale_factor
    original_height = height
    height = height * VTON_HEIGHT_EXPAND_FACTOR
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)
    # slice out only the "real" latents according to CatVTON paper
    real_latents = latents[:, :, :original_height * 2, :]
    return real_latents


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    height = height * VTON_HEIGHT_EXPAND_FACTOR
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    )

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
        latent_image_ids.shape
    )

    latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
    latent_image_ids = latent_image_ids.reshape(
        batch_size,
        latent_image_id_height * latent_image_id_width,
        latent_image_id_channels,
    )

    return latent_image_ids.to(device=device, dtype=dtype)
