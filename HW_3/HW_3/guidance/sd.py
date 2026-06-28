from diffusers import DDIMScheduler, StableDiffusionPipeline

import torch
import torch.nn as nn


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] loading stable diffusion...')

        model_key = "Manojb/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.t_range = t_range
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings
    
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=100):
        latent_model_input = torch.cat([latents_noisy] * 2)
            
        tt = torch.cat([t] * 2)
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred


    def get_sds_loss(
        self, 
        latents,
        text_embeddings, 
        guidance_scale=100, 
        grad_scale=1,
    ):
        batch_size = latents.shape[0]
        device = latents.device

        # sample random timestep t
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            (batch_size,),
            dtype=torch.long,
            device=device,
        )

        # sample noise epsilon
        noise = torch.randn_like(latents)

        # create noisy latent x_t
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        # predict noise with frozen diffusion model + classifier-free guidance
        with torch.no_grad():
            noise_pred = self.get_noise_preds(
                latents_noisy,
                t,
                text_embeddings,
                guidance_scale=guidance_scale,
            )

        # SDS gradient: epsilon_theta(x_t, t, c) - epsilon
        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)
        grad = grad_scale * w * (noise_pred - noise)

        # avoid NaNs/Infs
        grad = torch.nan_to_num(grad)

        # "fake loss" whose gradient w.r.t. latents is grad
        target = (latents - grad).detach()
        loss = 0.5 * torch.nn.functional.mse_loss(
            latents.float(),
            target.float(),
            reduction="sum",
        ) / batch_size

        return loss

    def get_pds_loss(
            self, src_latents, tgt_latents,
            src_text_embedding, tgt_text_embedding,
            guidance_scale=7.5,
            grad_scale=1,
    ):
        batch_size = tgt_latents.shape[0]
        device = tgt_latents.device

        # sample timestep t, and use t-1 as the previous denoising step
        t = torch.randint(
            self.min_step + 1,
            self.max_step + 1,
            (batch_size,),
            dtype=torch.long,
            device=device,
        )
        t_prev = t - 1

        # shared noises for source and target
        noise_t = torch.randn_like(tgt_latents)
        noise_prev = torch.randn_like(tgt_latents)

        # x_t and x_{t-1} for source and target, using shared noise
        src_xt = self.scheduler.add_noise(src_latents, noise_t, t)
        tgt_xt = self.scheduler.add_noise(tgt_latents, noise_t, t)

        src_xt_prev = self.scheduler.add_noise(src_latents, noise_prev, t_prev)
        tgt_xt_prev = self.scheduler.add_noise(tgt_latents, noise_prev, t_prev)

        # predict eps_theta(x_t, t, c)
        with torch.no_grad():
            src_eps = self.get_noise_preds(
                src_xt,
                t,
                src_text_embedding,
                guidance_scale=guidance_scale,
            )

            tgt_eps = self.get_noise_preds(
                tgt_xt,
                t,
                tgt_text_embedding,
                guidance_scale=guidance_scale,
            )

        def compute_z(x_t, x_t_prev, eps_pred, t, t_prev):
            alpha_t = self.alphas[t].view(batch_size, 1, 1, 1)
            alpha_prev = self.alphas[t_prev].view(batch_size, 1, 1, 1)

            beta_t = 1.0 - alpha_t
            beta_prev = 1.0 - alpha_prev

            # predicted x_0 from x_t
            pred_x0 = (x_t - beta_t.sqrt() * eps_pred) / alpha_t.sqrt()

            # DDIM/DDPM-style mean for x_{t-1}
            mu = alpha_prev.sqrt() * pred_x0 + beta_prev.sqrt() * eps_pred

            # stochastic latent z
            sigma = beta_prev.sqrt()
            z = (x_t_prev - mu) / (sigma + 1e-8)

            return z

        src_z = compute_z(src_xt, src_xt_prev, src_eps, t, t_prev).detach()
        tgt_z = compute_z(tgt_xt, tgt_xt_prev, tgt_eps, t, t_prev)

        loss = torch.nn.functional.mse_loss(
            tgt_z.float(),
            src_z.float(),
            reduction="mean",
        )

        return grad_scale * loss
    
    
    @torch.no_grad()
    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents
