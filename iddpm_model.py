from model.iddpm import *
from tool.base import normal_kl,meanflat,discretized_gaussian_log_likelihood
from math import  log as ln
NAT = 1. / ln(2)

class LearnedGaussianDiffusion(GaussianDiffusion):
    def __init__(
        self,
        model,
        vb_loss_weight = 0.001,  # lambda was 0.001 in the paper
        *args,
        **kwargs
    ):
        super().__init__(model, *args, **kwargs)
        assert model.out_dim == (
                    model.channels * 2), 'dimension out of unet must be twice the number of channels for learned variance - you can also set the `learned_variance` keyword argument on the Unet to be `True`'
        assert not model.self_condition, 'not supported yet'

        self.vb_loss_weight = vb_loss_weight

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t)
        model_output, pred_variance = model_output.chunk(2, dim = 1)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        x_start = maybe_clip(x_start)

        return ModelPrediction(pred_noise, x_start, pred_variance)

    def p_mean_variance(self, *, x, t, clip_denoised, model_output = None, **kwargs):
        model_output = default(model_output, lambda: self.model(x, t))
        pred_noise, var_interp_frac_unnormalized = model_output.chunk(2, dim = 1)

        min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
        max_log = extract(torch.log(self.betas), t, x.shape)
        var_interp_frac = unnormalize_to_zero_to_one(var_interp_frac_unnormalized)

        model_log_variance = var_interp_frac * max_log + (1 - var_interp_frac) * min_log
        model_variance = model_log_variance.exp()

        x_start = self.predict_start_from_noise(x, t, pred_noise)

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, _, _ = self.q_posterior(x_start, x, t)

        return model_mean, model_variance, model_log_variance, x_start

    def p_losses(self, x_start, t, noise = None, clip_denoised = False):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # model output

        model_output = self.model(x_t, t)

        # calculating kl loss for learned variance (interpolation)

        true_mean, _, true_log_variance_clipped = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        model_mean, _, model_log_variance, _ = self.p_mean_variance(x=x_t, t=t, clip_denoised=clip_denoised,
                                                                    model_output=model_output)

        # kl loss with detached model predicted mean, for stability reasons as in paper

        detached_model_mean = model_mean.detach()

        kl = normal_kl(true_mean, true_log_variance_clipped, detached_model_mean, model_log_variance)
        kl = meanflat(kl) * NAT

        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means=detached_model_mean,
                                                           log_scales=0.5 * model_log_variance)
        decoder_nll = meanflat(decoder_nll) * NAT

        # at the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))

        vb_losses = torch.where(t == 0, decoder_nll, kl)

        # simple loss - predicting noise, x0, or x_prev

        pred_noise, _ = model_output.chunk(2, dim=1)

        simple_losses = F.mse_loss(pred_noise, noise)

        return simple_losses + vb_losses.mean() * self.vb_loss_weight

