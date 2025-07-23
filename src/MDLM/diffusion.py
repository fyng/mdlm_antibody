import itertools
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from . import noise_schedule
from .models import DIT, ExponentialMovingAverage

class Diffusion(nn.Module):
    def __init__(
        self,
        config,
        tokenizer: AutoTokenizer,
        device: str,
    ):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.sampler = self.config.sampling.predictor
        self.gen_ppl_eval_model_name_or_path = self.config.eval.gen_ppl_eval_model_name_or_path
        if (not hasattr(self.tokenizer, 'mask_token')
            or self.tokenizer.mask_token is None):
            self.mask_index = self.vocab_size
            self.vocab_size += 1
        else:
            self.mask_index = self.tokenizer.mask_token_id
        self.parameterization = self.config.parameterization
        if self.config.backbone == 'dit':
            self.backbone = DIT(self.config, vocab_size=self.vocab_size)
        elif self.config.backbone == 'hf_dit':
            self.backbone = AutoModelForMaskedLM.from_pretrained(
            config.eval.checkpoint_path, trust_remote_code=True)
        else:
            raise ValueError(f'Unknown backbone: {self.config.backbone}')

        self.T = self.config.T
        self.subs_masking = self.config.subs_masking

        self.softplus = torch.nn.Softplus()
        self.dtype = torch.float32

        self.noise = noise_schedule.get_noise(self.config, dtype=self.dtype)
        if self.config.training.ema > 0:
            self.ema = ExponentialMovingAverage(
                itertools.chain(self.backbone.parameters(), self.noise.parameters()),
                decay=self.config.training.ema)
        else:
            self.ema = None
        
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.time_conditioning
        self.neg_infinity = -1000000.0


    def forward(self, x, sigma):
        sigma = self._process_sigma(sigma)
        with torch.amp.autocast(self.device, dtype=torch.float32):
            logits = self.backbone(x, sigma)
        
        if self.parameterization == 'subs':
            return self._subs_parameterization(logits=logits, xt=x)
        else:
            raise NotImplementedError(
                f'Unknown parameterization: {self.parameterization}'
            )
        
        return logits


    def _sample_prior(self, *batch_dims):
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)
    
    
    def _sample_categorical(self, categorical_probs):
        gumbel_norm = (
            1e-10
            - (torch.rand_like(categorical_probs) + 1e-10).log())
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    
    
    def _ddpm_caching_update(self, x, t, dt, p_x0=None):
        assert self.config.noise.type == 'loglinear'
        sigma_t, _ = self.noise(t)
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        assert move_chance_t.ndim == 3, move_chance_t.shape
        if p_x0 is None:
            p_x0 = self.forward(x, sigma_t).exp()
        
        assert move_chance_t.ndim == p_x0.ndim
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = self._sample_categorical(q_xs)
        
        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x
    
    
    def _ddpm_update(self, x, t, dt):
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        unet_conditioning = sigma_t
        log_p_x0 = self.forward(x, unet_conditioning)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        q_xs = log_p_x0.exp() * (move_chance_t
                                - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = self._sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * _x


    def get_score(self, x, sigma):
        model_output = self.forward(x, sigma)
        if self.parameterization == 'subs':
        # score(x, t) = p_t(y) / p_t(x)
        # => log score(x, t) = log p_t(y) - log p_t(x)
        
        # case 1: x = masked
        #   (i) y = unmasked
        #     log score(x, t) = log p_\theta(x)|_y + log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        #   (ii) y = masked
        #     log score(x, t) = 0

        # case 2: x = unmasked
        #   (i) y != masked, y != x
        #     log score(x_i, t) = - inf
        #   (ii) y = x 
        #     log score(x_i, t) = 0
        #   (iii) y = masked token
        #     log score(x_i, t) = - log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
            log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
            assert log_k.ndim == 1
            
            masked_score = model_output + log_k[:, None, None]
            masked_score[:, :, self.mask_index] = 0

            unmasked_score = self.neg_infinity * torch.ones_like(
                model_output)
            unmasked_score = torch.scatter(
                unmasked_score,
                -1,
                x[..., None],
                torch.zeros_like(unmasked_score[..., :1]))
            unmasked_score[:, :, self.mask_index] = - (
                log_k[:, None] * torch.ones_like(x))
            
            masked_indices = (x == self.mask_index).to(
                model_output.dtype)[:, :, None]
            model_output = (
                masked_score * masked_indices
                + unmasked_score * (1 - masked_indices))
            
        return model_output.exp()


    def _staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., self.mask_index] += extra_const
        return score
    
    
    def _unsqueeze(self, x, reference):
        return x.view(
            * x.shape,
            * ((1,) * (len(reference.shape) - len(x.shape))))
    
    
    def _transp_transition(self, i, sigma):
        sigma = self._unsqueeze(sigma, reference=i[..., None])
        edge = torch.exp(-sigma) * F.one_hot(
        i, num_classes=self.vocab_size)
        edge += torch.where(i == self.mask_index,
                            1 - torch.exp(-sigma).squeeze(-1),
                            0)[..., None]
        return edge


    def _analytic_update(self, x, t, step_size):
        curr_sigma, _ = self.noise(t)
        next_sigma, _ = self.noise(t - step_size)
        dsigma = curr_sigma - next_sigma
        score = self.get_score(x, curr_sigma)
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return self._sample_categorical(probs)


    def _denoiser_update(self, x, t):
        sigma, _ = self.noise(t)
        score = self.get_score(x, sigma)
        stag_score = self._staggered_score(score, sigma)
        probs = stag_score * self._transp_transition(x, sigma)
        probs[..., self.mask_index] = 0
        samples = self._sample_categorical(probs)
        return samples


    @torch.no_grad()
    def _sample_trajectory(self, num_steps=None, eps=1e-5):
        """Generate samples from the model."""
        batch_size_per_gpu = self.config.loader.eval_batch_size
        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self._sample_prior(
            batch_size_per_gpu,
            self.config.model.length
        ).to(self.device)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        samples = []
        ts = []
        
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == 'ddpm':
                x = self._ddpm_update(x, t, dt)
            elif self.sampler == 'ddpm_cache':
                p_x0_cache, x_next = self._ddpm_caching_update(
                x, t, dt, p_x0=p_x0_cache)
                if (not torch.allclose(x_next, x) or self.time_conditioning):
                    # Disable caching
                    p_x0_cache = None
                    x = x_next
            else:
                x = self._analytic_update(x, t, dt)
            samples.append(x)
            ts.append(t)

        # final denoising step
        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == 'analytic':
                x = self._denoiser_update(x, t)
            else:
                unet_conditioning = self.noise(t)[0]
                x = self.forward(x, unet_conditioning).argmax(dim=-1)
            samples.append(x)
            ts.append(t)
                
        return samples, ts
    
    
    @torch.no_grad()
    def _sample(self, num_steps=None, eps=1e-5):
        """Generate samples from the model."""
        batch_size_per_gpu = self.config.loader.eval_batch_size
        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self._sample_prior(
            batch_size_per_gpu,
            self.config.model.length
        ).to(self.device)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None
        
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == 'ddpm':
                x = self._ddpm_update(x, t, dt)
            elif self.sampler == 'ddpm_cache':
                p_x0_cache, x_next = self._ddpm_caching_update(
                x, t, dt, p_x0=p_x0_cache)
                if (not torch.allclose(x_next, x) or self.time_conditioning):
                    # Disable caching
                    p_x0_cache = None
                    x = x_next
            else:
                x = self._analytic_update(x, t, dt)

        # final denoising step
        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == 'analytic':
                x = self._denoiser_update(x, t)
            else:
                unet_conditioning = self.noise(t)[0]
                x = self.forward(x, unet_conditioning).argmax(dim=-1)

        return x
    
    
    def _process_sigma(self, sigma):
        if sigma is None:
            assert self.parameterization == 'ar'
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
            assert sigma.ndim == 1, sigma.shape
            
        return sigma
    
    
    def _subs_parameterization(self, logits, xt):
        # log prob at the mask index = - infinity
        logits[:, :, self.mask_index] += self.neg_infinity
        
        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits
    
    
    def _q_xt(
        self,         
        move_chance,
        batch: torch.Tensor,
    ):
        """
        Calculate noisy sample at x_t 
        """
        move_indices = torch.rand(* batch.shape, device=batch.device) < move_chance
        return torch.where(move_indices, self.mask_index, batch)
    
    
    def _subs_continuous(
        self, 
        x0: torch.Tensor, 
        model_output: torch.Tensor,
        sigma,
        dsigma,
    ) -> torch.Tensor:
        # TODO: check this is correct
        # SUBS parameterization, continuous time.
        log_p_theta = torch.gather(
            input=model_output,
            dim=-1,
            index=x0[:, :, None].type(torch.long) # expect int64 for index
        ).squeeze(-1)
        
        # # importance sampling correction
        # return log_p_theta * torch.log1p(-torch.exp(-self.noise.sigma_min))
        
        return - log_p_theta * (dsigma / torch.expm1(sigma))[:, None]
    
    
    def restore_model_and_sample(self, num_steps, eps=1e-5, store_traj=False):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        if self.ema:
            self.ema.store(itertools.chain(
                self.backbone.parameters(),
                self.noise.parameters())
            )
            self.ema.copy_to(itertools.chain(
                self.backbone.parameters(),
                self.noise.parameters())
            )
        self.backbone.eval()
        self.noise.eval()
        if store_traj:
            samples, timesteps = self._sample_trajectory(num_steps=num_steps, eps=eps)
        else:
            samples = self._sample(num_steps=num_steps, eps=eps)
        if self.ema:
            self.ema.restore(itertools.chain(
            self.backbone.parameters(),
            self.noise.parameters()))
        self.backbone.train()
        self.noise.train()
        
        if store_traj:
            return samples, timesteps
        
        return samples
    
    
    def train(self, mode: bool = True):
        self.backbone.train(mode)
        self.noise.train(mode)
        return self


    def eval(self):
        self.backbone.eval()
        self.noise.eval()
        return self
        
        
    def forward_pass_diffusion(self, batch):
        bsz = batch.shape[0]
        eps_t = torch.rand(bsz, device=batch.device)

        # Low discrepency sampler aka antithetic_sampling (appendix C3 in paper)
        offset = torch.arange(bsz, device=self.device) / bsz
        eps_t = (eps_t / bsz + offset) % 1
        t = (1 - self.sampling_eps) * eps_t + self.sampling_eps
        
        # # importance smapling
        # t = self.noise.importance_sampling_transformation(eps_t)
        # # TODO: add finite T timestep 
    
        sigma, dsigma = self.noise(t)
        conditioning = sigma[:, None]
        move_chance = 1 - torch.exp(-sigma[:, None]) # 1 - alpha_t
        
        batch_xt = self._q_xt(move_chance, batch)
        batch_logits = self.forward(batch_xt, conditioning)
        loss = self._subs_continuous(batch, batch_logits, sigma, dsigma)
        
        return loss
    
    
    def save(self, path):
        """Save the model to a file."""
        state_dict = {
            'backbone': self.backbone.state_dict(),
            'noise': self.noise.state_dict(),
            'config': self.config,
        }
        torch.save(state_dict, path)
        print(f"Model saved to {path}")
        
        
    @classmethod
    def load(cls, path, tokenizer, device):
        """Load the model from a file."""
        state_dict = torch.load(path, map_location=device)
        config = state_dict['config']
        
        # Create new instance with loaded config
        model = cls(config, tokenizer, device)
        
        # Load the state dictionaries
        model.backbone.load_state_dict(state_dict['backbone'])
        model.noise.load_state_dict(state_dict['noise'])
        
        print(f"Model loaded from {path}")
        return model
        
        