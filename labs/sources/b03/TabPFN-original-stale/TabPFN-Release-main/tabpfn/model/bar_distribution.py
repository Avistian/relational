import typing as tp
import torch
from torch import nn

from tabpfn.utils import print_once


# TODO: Merge functionality from BarDistribution and FullSupportBarDistribution
class BarDistribution(nn.Module):
    def __init__(self, borders: torch.Tensor, ignore_nan_targets=True):
        """
        Loss for a distribution over bars. The bars are defined by the borders. The loss is the negative log density of the
        distribution. The density is defined as a softmax over the logits, where the softmax is scaled by the width of the
        bars. This means that the density is 0 outside of the borders and the density is 1 on the borders.

        :param borders: tensor of shape (num_bars + 1) with the borders of the bars. here borders should start with min and end with max, where all values lie in (min,max) and are sorted
        :param ignore_nan_targets: if True, nan targets will be ignored, if False, an error will be raised
        """
        super().__init__()
        assert len(borders.shape) == 1
        borders = borders.contiguous()
        self.register_buffer("borders", borders)
        full_width = self.bucket_widths.sum()

        assert (
            1 - (full_width / (self.borders[-1] - self.borders[0]))
        ).abs() < 1e-2, f"diff: {full_width - (self.borders[-1] - self.borders[0])} with {full_width} {self.borders[-1]} {self.borders[0]}"
        assert (
            self.bucket_widths >= 0.0
        ).all(), "Please provide sorted borders!"  # This also allows size zero buckets
        self.ignore_nan_targets = ignore_nan_targets
        self.to(borders.device)

    @property
    def bucket_widths(self):
        return self.borders[1:] - self.borders[:-1]

    @property
    def num_bars(self):
        return len(self.borders) - 1

    def cdf(self, logits, ys):
        """
        Calculates the cdf of the distribution described by the logits. The cdf is scaled by the width of the bars.
        :param logits: tensor of shape (batch_size, ..., num_bars) with the logits describing the distribution
        :param ys: tensor of shape (batch_size, ..., num ys to evaluate) or shape (num ys to evaluate) with the targets
        """
        if len(ys.shape) < len(logits.shape) and len(ys.shape) == 1:
            # bring new borders to the same dim as logits up to the last dim
            ys = ys.repeat((logits.shape[:-1] + (1,)))
        else:
            assert (
                ys.shape[:-1] == logits.shape[:-1]
            ), f"ys.shape: {ys.shape} logits.shape: {logits.shape}"
        probs = torch.softmax(logits, dim=-1)
        buckets_of_ys = self.map_to_bucket_idx(ys).clamp(0, self.num_bars - 1)

        prob_so_far = torch.cumsum(probs, dim=-1) - probs
        prob_left_of_bucket = prob_so_far.gather(-1, buckets_of_ys)

        share_of_bucket_left = (
            (ys - self.borders[buckets_of_ys]) / self.bucket_widths[buckets_of_ys]
        ).clamp(0.0, 1.0)
        prob_in_bucket = probs.gather(-1, buckets_of_ys) * share_of_bucket_left

        prob_left_of_ys = prob_left_of_bucket + prob_in_bucket

        # just to fix numerical inaccuracies, if we had *exact* computation above we would not need the following:
        prob_left_of_ys[ys <= self.borders[0]] = 0.0
        prob_left_of_ys[ys >= self.borders[-1]] = 1.0
        assert not torch.isnan(prob_left_of_ys).any()

        return prob_left_of_ys.clip(0.0, 1.0)

    def get_probs_for_different_borders(self, logits, new_borders):
        """
        The logits describe the density of the distribution over the current self.borders. This function returns the logits
        if the self.borders were changed to new_borders. This is useful to average the logits of different models.
        """
        if (len(self.borders) == len(new_borders)) and (
            self.borders == new_borders
        ).all():
            return logits.softmax(-1)

        prob_left_of_borders = self.cdf(logits, new_borders)
        prob_left_of_borders[..., 0] = 0.0
        prob_left_of_borders[..., -1] = 1.0

        prob_mass_of_buckets = (
            prob_left_of_borders[..., 1:] - prob_left_of_borders[..., :-1]
        ).clip(min=0.0)

        return prob_mass_of_buckets

    def average_bar_distributions_into_this(
        self,
        list_of_bar_distributions: tp.List["BarDistribution"],
        list_of_logits: tp.List[torch.Tensor],
        average_logits: bool = False,
    ):
        """

        :param list_of_bar_distributions:
        :param list_of_logits:
        :param average_logits:
        :return:
        """
        probs = torch.stack(
            [
                bar_dist.get_probs_for_different_borders(l, self.borders)
                for bar_dist, l in zip(list_of_bar_distributions, list_of_logits)
            ],
            dim=0,
        )

        # print('old limits', self.borders, 'new limits', list_of_bar_distributions[0].borders)
        if average_logits:
            probs = probs.log().mean(dim=0).softmax(-1)
        else:
            probs = probs.mean(dim=0)
        assert not torch.isnan(
            probs.log()
        ).any(), f"probs: {probs[torch.isnan(probs.log())]}"
        return probs.log()

    def __setstate__(self, state):
        if "bucket_widths" in state:
            del state["bucket_widths"]
        super().__setstate__(state)
        self.__dict__.setdefault("append_mean_pred", False)

    def map_to_bucket_idx(self, y):
        # assert the borders are actually sorted
        assert (self.borders[1:] - self.borders[:-1] >= 0.0).all()
        target_sample = torch.searchsorted(self.borders, y) - 1
        target_sample[y == self.borders[0]] = 0
        target_sample[y == self.borders[-1]] = self.num_bars - 1
        return target_sample

    def ignore_init(self, y):
        ignore_loss_mask = torch.isnan(y)
        if ignore_loss_mask.any():
            if not self.ignore_nan_targets:
                raise ValueError(f"Found NaN in target {y}")
            print_once("A loss was ignored because there was nan target.")
        y[ignore_loss_mask] = self.borders[
            0
        ]  # this is just a default value, it will be ignored anyway
        return ignore_loss_mask

    def compute_scaled_log_probs(self, logits):
        # this is equivalent to log(p(y)) of the density p
        bucket_log_probs = torch.log_softmax(logits, -1)
        scaled_bucket_log_probs = bucket_log_probs - torch.log(self.bucket_widths)
        return scaled_bucket_log_probs

    def full_ce(self, logits, probs):
        return -(probs * torch.log_softmax(logits, -1)).sum(-1)

    def forward(
        self, logits, y, mean_prediction_logits=None
    ):  # gives the negative log density (the _loss_), y: T x B, logits: T x B x self.num_bars
        y = y.clone().view(*logits.shape[:-1])  # no trailing one dimension
        ignore_loss_mask = self.ignore_init(y)
        target_sample = self.map_to_bucket_idx(y)
        assert (target_sample >= 0).all() and (
            target_sample < self.num_bars
        ).all(), f"y {y} not in support set for borders (min_y, max_y) {self.borders}"
        assert (
            logits.shape[-1] == self.num_bars
        ), f"{logits.shape[-1]} vs {self.num_bars}"

        scaled_bucket_log_probs = self.compute_scaled_log_probs(logits)
        nll_loss = -scaled_bucket_log_probs.gather(
            -1, target_sample[..., None]
        ).squeeze(
            -1
        )  # T x B

        if mean_prediction_logits is not None:  # TO BE REMOVED AFTER BO SUBMISSION
            nll_loss = torch.cat(
                (nll_loss, self.mean_loss(logits, mean_prediction_logits)), 0
            )

        nll_loss[ignore_loss_mask] = 0.0
        return nll_loss

    def mean_loss(
        self, logits, mean_prediction_logits
    ):  # TO BE REMOVED AFTER BO SUBMISSION
        scaled_mean_log_probs = self.compute_scaled_log_probs(mean_prediction_logits)
        if not self.training:
            print("Calculating loss incl mean prediction loss for nonmyopic BO.")
        assert (len(logits.shape) == 3) and (len(scaled_mean_log_probs.shape) == 2), (
            len(logits.shape),
            len(scaled_mean_log_probs.shape),
        )
        means = self.mean(logits).detach()  # T x B
        target_mean = self.map_to_bucket_idx(means).clamp_(
            0, self.num_bars - 1
        )  # T x B
        return (
            -scaled_mean_log_probs.gather(1, target_mean.T).mean(1).unsqueeze(0)
        )  # 1 x B

    def mean(self, logits):
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        p = torch.softmax(logits, -1)
        return p @ bucket_means

    def median(self, logits):
        return self.icdf(logits, 0.5)

    def cdf_temporary(self, logits):
        """
        Cumulative distribution function
        TODO: this already exists here, make sure to merge, at the moment still used.
        :param logits:
        :return:
        """
        probs = logits.softmax(-1)
        cumprobs = -torch.cumsum(probs, -1).sum(axis=-1)
        return cumprobs

    def icdf(self, logits, left_prob):
        """
        Implementation of the quantile function
        :param logits: Tensor of any shape, with the last dimension being logits
        :param left_prob: float: The probability mass to the left of the result.
        :return: Position with `left_prob` probability weight to the left.
        """
        probs = logits.softmax(-1)
        cumprobs = torch.cumsum(probs, -1)
        idx = (
            torch.searchsorted(
                cumprobs,
                left_prob * torch.ones(*cumprobs.shape[:-1], 1, device=logits.device),
            )
            .squeeze(-1)
            .clamp(0, cumprobs.shape[-1] - 1)
        )  # this might not do the right for outliers
        cumprobs = torch.cat(
            [torch.zeros(*cumprobs.shape[:-1], 1, device=logits.device), cumprobs], -1
        )

        rest_prob = left_prob - cumprobs.gather(-1, idx[..., None]).squeeze(-1)
        left_border = self.borders[idx]
        right_border = self.borders[idx + 1]
        return left_border + (right_border - left_border) * rest_prob / probs.gather(
            -1, idx[..., None]
        ).squeeze(-1)

    def quantile(self, logits, center_prob=0.682):
        side_probs = (1.0 - center_prob) / 2
        return torch.stack(
            (self.icdf(logits, side_probs), self.icdf(logits, 1.0 - side_probs)), -1
        )

    def ucb(self, logits, best_f, rest_prob=(1 - 0.682) / 2, maximize=True):
        """
        UCB utility. Rest Prob is the amount of utility above (below) the confidence interval that is ignored.
        Higher rest_prob is equivalent to lower beta in the standard GP-UCB formulation.
        :param logits: Logits, as returned by the Transformer.
        :param rest_prob: The amount of utility above (below) the confidence interval that is ignored.
        The default is equivalent to using GP-UCB with `beta=1`.
        To get the corresponding `beta`, where `beta` is from
        the standard GP definition of UCB `ucb_utility = mean + beta * std`,
        you can use this computation: `beta = math.sqrt(2)*torch.erfinv(torch.tensor(2*(1-rest_prob)-1))`.
        :param maximize:
        :return: utility
        """
        if maximize:
            rest_prob = 1 - rest_prob
        return self.icdf(logits, rest_prob)

    def mode(self, logits):
        density = logits.softmax(-1) / self.bucket_widths
        mode_inds = density.argmax(-1)
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        return bucket_means[mode_inds]

    def ei(
        self, logits, best_f, maximize=True
    ):  # logits: evaluation_points x batch x feature_dim
        bucket_diffs = self.borders[1:] - self.borders[:-1]
        assert maximize
        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(logits[..., 0].shape, best_f, device=logits.device)

        best_f = best_f[..., None].repeat(*[1] * len(best_f.shape), logits.shape[-1])
        clamped_best_f = best_f.clamp(self.borders[:-1], self.borders[1:])

        # bucket_contributions = (best_f[...,None] < self.borders[:-1]).float() * bucket_means
        # true bucket contributions
        bucket_contributions = (
            (self.borders[1:] ** 2 - clamped_best_f**2) / 2
            - best_f * (self.borders[1:] - clamped_best_f)
        ) / bucket_diffs

        p = torch.softmax(logits, -1)
        return torch.einsum("...b,...b->...", p, bucket_contributions)

    def pi(
        self, logits, best_f, maximize=True
    ):  # logits: evaluation_points x batch x feature_dim
        """
        Acquisition Function: Probability of Improvement
        :param logits: as returned by Transformer
        :param best_f: best evaluation so far (the incumbent)
        :param maximize: whether to maximize
        :return: utility
        """
        assert maximize is True
        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(logits[..., 0].shape, best_f, device=logits.device)
        p = torch.softmax(logits, -1)
        border_widths = self.borders[1:] - self.borders[:-1]
        factor = 1.0 - ((best_f[..., None] - self.borders[:-1]) / border_widths).clamp(
            0.0, 1.0
        )
        return (p * factor).sum(-1)

    def mean_of_square(self, logits):
        """
        Computes E[x^2].
        :param logits: Output of the model.
        """
        left_borders = self.borders[:-1]
        right_borders = self.borders[1:]
        bucket_mean_of_square = (
            left_borders.square()
            + right_borders.square()
            + left_borders * right_borders
        ) / 3.0
        p = torch.softmax(logits, -1)
        return p @ bucket_mean_of_square

    def variance(self, logits):
        return self.mean_of_square(logits) - self.mean(logits).square()

    def plot(self, logits, ax=None, zoom_to_quantile=None, **kwargs):
        """
        Plots the distribution.
        :param logits: Output of the model.
        :param ax: Axes to plot on.
        :param kwargs: Additional arguments to pass to the plot
        """
        import matplotlib.pyplot as plt

        logits = logits.squeeze()
        assert logits.dim() == 1, "logits should be 1d, at least after squeezing."
        if ax is None:
            ax = plt.gca()
        if zoom_to_quantile is not None:
            lower_bounds, upper_bounds = self.quantile(
                logits, zoom_to_quantile
            ).transpose(0, -1)
            lower_bound = lower_bounds.min().item()
            upper_bound = upper_bounds.max().item()
            ax.set_xlim(lower_bound, upper_bound)
            border_mask = (self.borders[:-1] >= lower_bound) & (
                self.borders[1:] <= upper_bound
            )
        else:
            border_mask = slice(None)
        p = torch.softmax(logits, -1) / self.bucket_widths
        ax.bar(
            self.borders[:-1][border_mask],
            p[border_mask],
            self.bucket_widths[border_mask],
            **kwargs,
        )
        return ax


class FullSupportBarDistribution(BarDistribution):
    def __init__(
        self, borders, **kwargs
    ):  # here borders should start with min and end with max, where all values lie in (min,max) and are sorted
        """
        :param borders:
        """
        super().__init__(borders, **kwargs)
        self.assert_support(allow_zero_bucket_left=False)

        losses_per_bucket = torch.zeros_like(self.bucket_widths)
        self.register_buffer("losses_per_bucket", losses_per_bucket)

    def assert_support(self, allow_zero_bucket_left=False):
        if allow_zero_bucket_left:
            assert (
                self.bucket_widths[-1] > 0
            ), f"Half Normal weight must be greater than 0 (got -1:{self.bucket_widths[-1]})."
            # This fixes the distribution if the half normal at zero is width zero
            if self.bucket_widths[0] == 0:
                self.borders[0] = self.borders[0] - 1
                self.bucket_widths[0] = 1.0
        else:
            assert (
                self.bucket_widths[0] > 0 and self.bucket_widths[-1] > 0
            ), f"Half Normal weight must be greater than 0 (got 0: {self.bucket_widths[0]} and -1:{self.bucket_widths[-1]})."

    @staticmethod
    def halfnormal_with_p_weight_before(range_max, p=0.5):
        s = range_max / torch.distributions.HalfNormal(torch.tensor(1.0)).icdf(
            torch.tensor(p)
        )
        return torch.distributions.HalfNormal(s)

    def forward(self, logits, y, mean_prediction_logits=None):
        """
        Returns the negative log density (the _loss_), y: T x B, logits: T x B x self.num_bars

        :param logits: Tensor of shape T x B x self.num_bars
        :param y: Tensor of shape T x B
        :param mean_prediction_logits:
        :return:
        """
        assert self.num_bars > 1
        y = y.clone().view(*logits.shape[:-1])  # no trailing one dimension
        ignore_loss_mask = self.ignore_init(y)  # alters y
        target_sample = self.map_to_bucket_idx(y)  # shape: T x B (same as y)
        target_sample.clamp_(0, self.num_bars - 1)

        assert (
            logits.shape[-1] == self.num_bars
        ), f"{logits.shape[-1]} vs {self.num_bars}"
        assert (target_sample >= 0).all() and (
            target_sample < self.num_bars
        ).all(), f"y {y} not in support set for borders (min_y, max_y) {self.borders}"
        assert (
            logits.shape[-1] == self.num_bars
        ), f"{logits.shape[-1]} vs {self.num_bars}"
        # ignore all position with nan values

        scaled_bucket_log_probs = self.compute_scaled_log_probs(logits)

        assert len(scaled_bucket_log_probs) == len(target_sample), (
            len(scaled_bucket_log_probs),
            len(target_sample),
        )
        log_probs = scaled_bucket_log_probs.gather(
            -1, target_sample.unsqueeze(-1)
        ).squeeze(-1)

        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )

        log_probs[target_sample == 0] += side_normals[0].log_prob(
            (self.borders[1] - y[target_sample == 0]).clamp(min=0.00000001)
        ) + torch.log(self.bucket_widths[0])
        log_probs[target_sample == self.num_bars - 1] += side_normals[1].log_prob(
            (y[target_sample == self.num_bars - 1] - self.borders[-2]).clamp(
                min=0.00000001
            )
        ) + torch.log(self.bucket_widths[-1])

        nll_loss = -log_probs

        if mean_prediction_logits is not None:  # TO BE REMOVED AFTER BO PAPER IS DONE
            assert (
                not ignore_loss_mask.any()
            ), "Ignoring examples is not implemented with mean pred."
            if not torch.is_grad_enabled():
                print("Warning: loss is not correct in absolute terms.")
            nll_loss = torch.cat(
                (nll_loss, self.mean_loss(logits, mean_prediction_logits)), 0
            )

        if ignore_loss_mask.any():
            nll_loss[ignore_loss_mask] = 0.0

        # TODO: Check with samuel whether to keep
        self.losses_per_bucket += (
            torch.scatter(
                self.losses_per_bucket,
                0,
                target_sample[~ignore_loss_mask].flatten(),
                nll_loss[~ignore_loss_mask].flatten().detach(),
            )
            / target_sample[~ignore_loss_mask].numel()
        )

        return nll_loss

    def pdf(self, logits, y):
        """
        Probability density function at y.
        """
        return torch.exp(self.forward(logits, y))

    def sample(self, logits, t=1.0):
        """
        Samples values from the distribution.
        Temperature t
        """
        p_cdf = torch.rand(*logits.shape[:-1])
        return torch.tensor(
            [self.icdf(logits[i, :] / t, p) for i, p in enumerate(p_cdf.tolist())]
        )

    def mean(self, logits):
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        p = torch.softmax(logits, -1)
        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        bucket_means[0] = -side_normals[0].mean + self.borders[1]
        bucket_means[-1] = side_normals[1].mean + self.borders[-2]
        return p @ bucket_means.to(logits.device)

    def mean_of_square(self, logits):
        """
        Computes E[x^2].
        :param logits: Output of the model.
        """
        left_borders = self.borders[:-1]
        right_borders = self.borders[1:]
        bucket_mean_of_square = (
            left_borders.square()
            + right_borders.square()
            + left_borders * right_borders
        ) / 3.0
        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        bucket_mean_of_square[0] = (
            side_normals[0].variance
            + (-side_normals[0].mean + self.borders[1]).square()
        )
        bucket_mean_of_square[-1] = (
            side_normals[1].variance
            + (side_normals[1].variance + self.borders[-2]).square()
        )
        p = torch.softmax(logits, -1)
        return p @ bucket_mean_of_square

    def pi(
        self, logits, best_f, maximize=True
    ):  # logits: evaluation_points x batch x feature_dim
        """
        Acquisition Function: Probability of Improvement
        :param logits: as returned by Transformer (evaluation_points x batch x feature_dim)
        :param best_f: best evaluation so far (the incumbent)
        :param maximize: whether to maximize
        :return: utility
        """
        assert maximize is True
        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(
                logits[..., 0].shape, best_f, device=logits.device
            )  # evaluation_points x batch
        assert (
            best_f.shape == logits[..., 0].shape
        ), f"best_f.shape: {best_f.shape}, logits.shape: {logits.shape}"
        p = torch.softmax(logits, -1)  # evaluation_points x batch
        border_widths = self.borders[1:] - self.borders[:-1]
        factor = 1.0 - ((best_f[..., None] - self.borders[:-1]) / border_widths).clamp(
            0.0, 1.0
        )  # evaluation_points x batch x num_bars

        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        position_in_side_normals = (
            -(best_f - self.borders[1]).clamp(max=0.0),
            (best_f - self.borders[-2]).clamp(min=0.0),
        )  # evaluation_points x batch
        factor[..., 0] = 0.0
        factor[..., 0][position_in_side_normals[0] > 0.0] = side_normals[0].cdf(
            position_in_side_normals[0][position_in_side_normals[0] > 0.0]
        )
        factor[..., -1] = 1.0
        factor[..., -1][position_in_side_normals[1] > 0.0] = 1.0 - side_normals[1].cdf(
            position_in_side_normals[1][position_in_side_normals[1] > 0.0]
        )
        return (p * factor).sum(-1)

    def ei_for_halfnormal(self, scale, best_f, maximize=True):
        """
        This is the EI for a standard normal distribution with mean 0 and variance `scale` times 2.
        Which is the same as the half normal EI.
        I tested this with MC approximation:
        ei_for_halfnormal = lambda scale, best_f: (torch.distributions.HalfNormal(torch.tensor(scale)).sample((10_000_000,))- best_f ).clamp(min=0.).mean()
        print([(ei_for_halfnormal(scale,best_f), FullSupportBarDistribution().ei_for_halfnormal(scale,best_f)) for scale in [0.1,1.,10.] for best_f in [.1,10.,4.]])
        :param scale:
        :param best_f:
        :param maximize:
        :return:
        """
        assert maximize
        mean = torch.tensor(0.0)
        u = (mean - best_f) / scale
        normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
        try:
            ucdf = normal.cdf(u)
        except ValueError:
            print(f"u: {u}, best_f: {best_f}, scale: {scale}")
            raise
        updf = torch.exp(normal.log_prob(u))
        normal_ei = scale * (updf + u * ucdf)
        return 2 * normal_ei

    def ei(
        self, logits, best_f, maximize=True
    ):  # logits: evaluation_points x batch x feature_dim
        if torch.isnan(logits).any():
            raise ValueError(f"logits contains NaNs: {logits}")
        bucket_diffs = self.borders[1:] - self.borders[:-1]
        assert maximize
        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(logits[..., 0].shape, best_f, device=logits.device)
        assert (
            best_f.shape == logits[..., 0].shape
        ), f"best_f.shape: {best_f.shape}, logits.shape: {logits.shape}"

        best_f_per_logit = best_f[..., None].repeat(
            *[1] * len(best_f.shape), logits.shape[-1]
        )
        clamped_best_f = best_f_per_logit.clamp(self.borders[:-1], self.borders[1:])

        # true bucket contributions
        bucket_contributions = (
            (self.borders[1:] ** 2 - clamped_best_f**2) / 2
            - best_f_per_logit * (self.borders[1:] - clamped_best_f)
        ) / bucket_diffs

        # extra stuff for continuous
        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        position_in_side_normals = (
            -(best_f - self.borders[1]).clamp(max=0.0),
            (best_f - self.borders[-2]).clamp(min=0.0),
        )  # evaluation_points x batch

        bucket_contributions[..., -1] = self.ei_for_halfnormal(
            side_normals[1].scale, position_in_side_normals[1]
        )

        bucket_contributions[..., 0] = self.ei_for_halfnormal(
            side_normals[0].scale, torch.zeros_like(position_in_side_normals[0])
        ) - self.ei_for_halfnormal(side_normals[0].scale, position_in_side_normals[0])

        p = torch.softmax(logits, -1)
        return torch.einsum("...b,...b->...", p, bucket_contributions)


def get_bucket_limits(
    num_outputs: int,
    full_range: tuple = None,
    ys: torch.Tensor = None,
    verbose: bool = False,
    widen_bucket_limits_factor: float = None,
):
    """
    This function is used to automatically decide for a set of bucket limits based on a distritbution of ys.
    :param num_outputs: This is only tested for num_outputs=1, but should work for larger num_outputs as well.
    :param full_range: If ys is not passed, this is the range of the ys that should be used to estimate the bucket limits.
    :param ys: If ys is passed, this is the ys that should be used to estimate the bucket limits. Do not pass full_range in this case.
    :param verbose: If True, print some information.
    :param widen_bucket_limits_factor:  If set, the bucket limits are widened by this factor. This allows to have a slightly larger range than the actual data.
    """
    assert (ys is None) != (
        full_range is None
    ), "Either full_range or ys must be passed."

    if ys is not None:
        ys = ys.flatten()
        ys = ys[~torch.isnan(ys)]
        assert (
            len(ys) > num_outputs
        ), f"Number of ys :{len(ys)} must be larger than num_outputs: {num_outputs}"
        if len(ys) % num_outputs:
            ys = ys[: -(len(ys) % num_outputs)]
        print(
            f"ALL: Using {len(ys)} y evals to estimate {num_outputs} buckets. Cut off the last {len(ys) % num_outputs} ys."
        )
        ys_per_bucket = len(ys) // num_outputs
        if full_range is None:
            full_range = (ys.min(), ys.max())
        else:
            assert (
                full_range[0] <= ys.min() and full_range[1] >= ys.max()
            ), f"full_range {full_range} not in range of ys {ys.min(), ys.max()}"
            full_range = torch.tensor(full_range)
        ys_sorted, ys_order = ys.sort(0)
        bucket_limits = (
            ys_sorted[ys_per_bucket - 1 :: ys_per_bucket][:-1]
            + ys_sorted[ys_per_bucket::ys_per_bucket]
        ) / 2
        if verbose:
            print(
                f"ALL: Using {len(ys)} y evals to estimate {num_outputs} buckets. Cut off the last {len(ys) % num_outputs} ys."
            )
            print(full_range)
        bucket_limits = torch.cat(
            [full_range[0].unsqueeze(0), bucket_limits, full_range[1].unsqueeze(0)], 0
        )
        if widen_bucket_limits_factor is not None:
            print("ALL", bucket_limits[-1], bucket_limits[0])
            bucket_limits = bucket_limits * widen_bucket_limits_factor
            print("ALL: WIDENED", bucket_limits[-1], bucket_limits[0])

    else:
        class_width = (full_range[1] - full_range[0]) / num_outputs
        bucket_limits = torch.cat(
            [
                full_range[0] + torch.arange(num_outputs).float() * class_width,
                torch.tensor(full_range[1]).unsqueeze(0),
            ],
            0,
        )

    assert (
        len(bucket_limits) - 1 == num_outputs
    ), f"len(bucket_limits) - 1 == {len(bucket_limits) - 1} != {num_outputs} == num_outputs"
    if not widen_bucket_limits_factor or widen_bucket_limits_factor == 1.0:
        assert (
            full_range[0] == bucket_limits[0]
        ), f"{full_range[0]} != {bucket_limits[0]}"
        assert (
            full_range[-1] == bucket_limits[-1]
        ), f"{full_range[-1]} != {bucket_limits[-1]}"

    return bucket_limits


def get_custom_bar_dist(borders, criterion):
    # Tested that a bar_dist with borders 0.54 (-> softplus 1.0) yields the same bar distribution as the passed one.
    borders_ = torch.nn.functional.softplus(borders) + 0.001
    borders_ = torch.cumsum(
        torch.cat([criterion.borders[0:1], criterion.bucket_widths]) * borders_, 0
    )
    criterion_ = criterion.__class__(
        borders=borders_, handle_nans=criterion.handle_nans
    )
    return criterion_
