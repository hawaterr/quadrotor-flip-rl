"""Minimal single-task TD-MPC2 (Hansen et al., 2024), CPU friendly.

Kept from the paper: SimNorm latent, latent dynamics + reward + Q-ensemble trained with a multi-step
consistency loss, discrete (two-hot, symlog) regression for reward and value, a max-entropy policy prior,
and MPPI planning in latent space. Dropped: everything multi-task (task embeddings, action masks).
Added: a termination head, since QuadEnv ends episodes on a crash (the original assumes fixed-length episodes).
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import Config


# --- math helpers ----------------------------------------------------------------------------------

def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


class TwoHot:
    """Scalars <-> categorical over num_bins evenly spaced bins in symlog space."""

    def __init__(self, cfg):
        self.vmin, self.vmax, self.n = cfg.vmin, cfg.vmax, cfg.num_bins
        self.delta = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
        self.centers = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins)

    def encode(self, x):
        """(..., 1) scalars -> (..., num_bins) soft labels."""
        x = symlog(x).clamp(self.vmin, self.vmax).squeeze(-1)
        pos = (x - self.vmin) / self.delta
        lo = pos.floor().long().clamp(0, self.n - 1)
        hi = (lo + 1).clamp(max=self.n - 1)
        frac = (pos - lo).unsqueeze(-1)
        target = torch.zeros(*x.shape, self.n)
        target.scatter_(-1, lo.unsqueeze(-1), 1 - frac)
        target.scatter_add_(-1, hi.unsqueeze(-1), frac)
        return target

    def decode(self, logits):
        """(..., num_bins) logits -> (..., 1) scalars."""
        return symexp((F.softmax(logits, -1) * self.centers).sum(-1, keepdim=True))

    def loss(self, logits, x):
        """Cross-entropy between logits and the two-hot encoding of x, shape (..., 1)."""
        return -(self.encode(x) * F.log_softmax(logits, -1)).sum(-1, keepdim=True)


class SimNorm(nn.Module):
    """Softmax over groups of `dim` features: keeps the latent on a product of simplices."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        shape = x.shape
        return F.softmax(x.view(*shape[:-1], -1, self.dim), -1).view(shape)


def mlp(in_dim, hidden, out_dim, out_act=None, dropout=0.0):
    """Linear -> LayerNorm -> SiLU blocks. With out_act, the last layer also gets LayerNorm + out_act.
    (The paper uses Mish; SiLU is near-identical and ~2x faster on CPU.)"""
    layers, d = [], in_dim
    for i, h in enumerate(hidden):
        layers += [nn.Linear(d, h)]
        if dropout and i == 0:
            layers += [nn.Dropout(dropout)]
        layers += [nn.LayerNorm(h), nn.SiLU()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    if out_act is not None:
        layers += [nn.LayerNorm(out_dim), out_act]
    return nn.Sequential(*layers)


class RunningScale:
    """Running estimate of the 5-95 percentile range of Q, used to normalise the policy loss."""

    def __init__(self, tau):
        self.tau, self.value = tau, torch.tensor(1.0)

    def update(self, x):
        lo, hi = torch.quantile(x.detach().flatten(), torch.tensor([0.05, 0.95]))
        self.value.lerp_(hi - lo, self.tau)

    def __call__(self, x):
        return x / self.value.clamp(min=1.0)


# --- world model -----------------------------------------------------------------------------------

class WorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, cfg):
        super().__init__()
        self.cfg = cfg
        z, a, h = cfg.latent_dim, act_dim, [cfg.mlp_dim] * 2
        self.encoder = mlp(obs_dim, [cfg.enc_dim], z, SimNorm(cfg.simnorm_dim))
        self.dynamics = mlp(z + a, h, z, SimNorm(cfg.simnorm_dim))
        self.reward = mlp(z + a, h, cfg.num_bins)
        self.termination = mlp(z, h, 1)  # logit that the episode ended on arriving at z
        self.pi = mlp(z, h, 2 * a)
        self.qs = nn.ModuleList([mlp(z + a, h, cfg.num_bins, dropout=cfg.dropout) for _ in range(cfg.num_q)])
        # Zero-init reward and Q outputs: predictions start at 0 instead of random
        for head in [self.reward, *self.qs]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.target_qs = nn.ModuleList([mlp(z + a, h, cfg.num_bins, dropout=cfg.dropout) for _ in range(cfg.num_q)])
        self.target_qs.load_state_dict(self.qs.state_dict())
        self.target_qs.requires_grad_(False)

    def next(self, z, a):
        return self.dynamics(torch.cat([z, a], -1))

    def policy(self, z):
        """Sample a squashed Gaussian action. Returns (mean, action, log_prob)."""
        mean, log_std = self.pi(z).chunk(2, -1)
        lo, hi = self.cfg.log_std_min, self.cfg.log_std_max
        log_std = lo + 0.5 * (hi - lo) * (torch.tanh(log_std) + 1)
        eps = torch.randn_like(mean)
        action = mean + eps * log_std.exp()
        log_prob = (-0.5 * eps.pow(2) - log_std - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
        mean, action = torch.tanh(mean), torch.tanh(action)
        log_prob = log_prob - torch.log(F.relu(1 - action.pow(2)) + 1e-6).sum(-1, keepdim=True)
        return mean, action, log_prob

    def q(self, z, a, two_hot, mode="min", target=False):
        """mode="all": logits of every head; "min"/"avg": min/mean value of two random heads."""
        za = torch.cat([z, a], -1)
        heads = self.target_qs if target else self.qs
        if mode == "all":
            return torch.stack([q(za) for q in heads])
        i, j = np.random.choice(len(heads), 2, replace=False)
        q1, q2 = two_hot.decode(heads[i](za)), two_hot.decode(heads[j](za))
        return torch.min(q1, q2) if mode == "min" else (q1 + q2) / 2


# --- agent -----------------------------------------------------------------------------------------

class TDMPC2:
    def __init__(self, obs_dim, act_dim, cfg):
        """cfg: the `agent` section of configs/tdmpc2.yaml."""
        self.cfg = cfg
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.model = WorldModel(obs_dim, act_dim, self.cfg)
        self.two_hot = TwoHot(self.cfg)
        self.scale = RunningScale(self.cfg.tau)
        m, c = self.model, self.cfg
        self.optim = torch.optim.Adam([
            {"params": m.encoder.parameters(), "lr": c.lr * c.enc_lr_scale},
            {"params": [*m.dynamics.parameters(), *m.reward.parameters(),
                        *m.termination.parameters(), *m.qs.parameters()]},
        ], lr=c.lr)
        self.pi_optim = torch.optim.Adam(m.pi.parameters(), lr=c.lr, eps=1e-5)
        self.prev_mean = torch.zeros(c.horizon, act_dim)  # MPPI warm start

    def n_params(self):
        return sum(p.numel() for name, p in self.model.named_parameters() if not name.startswith("target_"))

    # --- acting ---

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, plan=True):
        """t0: first step of an episode (resets the MPPI warm start). plan=False uses the policy prior only."""
        self.model.eval()
        z = self.model.encoder(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        if plan:
            a = self._plan(z, t0, eval_mode)
        else:
            mean, a, _ = self.model.policy(z)
            a = (mean if eval_mode else a)[0]
        return a.clamp(-1, 1).numpy()

    def _estimate_value(self, z, actions):
        """Discounted return of action sequences (H, N, A) from latent z (N, Z), plus a Q bootstrap."""
        G, discount, done = 0, 1, torch.zeros(z.shape[0], 1)
        for a in actions:
            r = self.two_hot.decode(self.model.reward(torch.cat([z, a], -1)))
            z = self.model.next(z, a)
            G = G + discount * (1 - done) * r
            discount *= self.cfg.discount
            done = torch.clamp(done + (self.model.termination(z) > 0).float(), max=1)
        _, a, _ = self.model.policy(z)
        return G + discount * (1 - done) * self.model.q(z, a, self.two_hot, "avg")

    def _plan(self, z, t0, eval_mode):
        c, A = self.cfg, self.act_dim
        H, N, P = c.horizon, c.num_samples, c.num_pi_trajs
        actions = torch.empty(H, N, A)
        # Trajectories from the policy prior
        _z = z.repeat(P, 1)
        for t in range(H):
            actions[t, :P] = self.model.policy(_z)[1]
            _z = self.model.next(_z, actions[t, :P])
        z = z.repeat(N, 1)
        mean, std = torch.zeros(H, A), torch.full((H, A), c.max_std)
        if not t0:
            mean[:-1] = self.prev_mean[1:]
        for _ in range(c.iterations):
            actions[:, P:] = (mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(H, N - P, A)).clamp(-1, 1)
            value = self._estimate_value(z, actions).nan_to_num(0).squeeze(1)
            elite = torch.topk(value, c.num_elites).indices
            elite_value, elite_actions = value[elite], actions[:, elite]
            score = torch.exp(c.temperature * (elite_value - elite_value.max()))
            score = (score / score.sum()).view(1, -1, 1)
            mean = (score * elite_actions).sum(1)
            std = ((score * (elite_actions - mean.unsqueeze(1)) ** 2).sum(1)).sqrt().clamp(c.min_std, c.max_std)
        # Pick one elite trajectory in proportion to its score, and execute its first action
        idx = torch.multinomial(score.view(-1), 1).item()
        a = elite_actions[0, idx]
        if not eval_mode:
            a = a + std[0] * torch.randn(A)
        self.prev_mean = mean
        return a

    # --- learning ---

    def update(self, obs, action, reward, terminated):
        """obs (H+1, B, O), action/reward/terminated (H, B, ·). Returns a dict of losses."""
        c, m = self.cfg, self.model
        m.train()
        with torch.no_grad():
            next_z = m.encoder(obs[1:])
            _, next_a, _ = m.policy(next_z)
            q_next = m.q(next_z, next_a, self.two_hot, "min", target=True)
            td_target = reward + c.discount * (1 - terminated) * q_next

        # Roll the latent dynamics forward from the first observation only
        z = m.encoder(obs[0])
        zs = [z]
        consistency = 0
        for t in range(c.horizon):
            z = m.next(z, action[t])
            consistency = consistency + c.rho ** t * F.mse_loss(z, next_z[t])
            zs.append(z)
        zs = torch.stack(zs)
        _zs = zs[:-1]
        qs = m.q(_zs, action, self.two_hot, "all")  # (num_q, H, B, bins)
        reward_logits = m.reward(torch.cat([_zs, action], -1))
        term_logits = m.termination(zs[1:])

        rho = (c.rho ** torch.arange(c.horizon, dtype=torch.float32)).view(-1, 1, 1)
        reward_loss = (rho * self.two_hot.loss(reward_logits, reward)).mean(dim=(1, 2)).sum()
        value_loss = (rho * self.two_hot.loss(qs, td_target.expand(c.num_q, -1, -1, -1))).mean(dim=(2, 3)).sum()
        term_loss = F.binary_cross_entropy_with_logits(term_logits, terminated)
        loss = (c.consistency_coef * consistency + c.reward_coef * reward_loss
                + c.value_coef * value_loss) / c.horizon + c.termination_coef * term_loss
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(m.parameters(), c.grad_clip)
        self.optim.step()

        pi_loss = self._update_pi(zs.detach())

        with torch.no_grad():
            for p, tp in zip(m.qs.parameters(), m.target_qs.parameters()):
                tp.lerp_(p, c.tau)
        return {"consistency": consistency.item(), "reward": reward_loss.item(), "value": value_loss.item(),
                "termination": term_loss.item(), "pi": pi_loss, "grad_norm": grad_norm.item(),
                "q_scale": float(self.scale.value)}

    def _update_pi(self, zs):
        c, m = self.cfg, self.model
        _, a, log_prob = m.policy(zs)
        q = m.q(zs, a, self.two_hot, "avg")  # gradients flow into the action only; Q is not stepped here
        self.scale.update(q[0])
        q = self.scale(q)
        rho = (c.rho ** torch.arange(len(q), dtype=torch.float32)).view(-1, 1, 1)
        loss = (rho * (c.entropy_coef * log_prob - q)).mean(dim=(1, 2)).mean()
        self.pi_optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(m.pi.parameters(), c.grad_clip)
        self.pi_optim.step()
        return loss.item()

    # --- checkpoints ---

    def save(self, path, **meta):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model": self.model.state_dict(), "cfg": self.cfg.to_dict(), "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim, "scale": self.scale.value, **meta}, path)

    @classmethod
    def load(cls, path):
        """Returns (agent, checkpoint dict); the dict also holds whatever meta was passed to save()."""
        ckpt = torch.load(path, weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["act_dim"], Config(ckpt["cfg"]))
        agent.model.load_state_dict(ckpt["model"])
        agent.scale.value = ckpt["scale"]
        return agent, ckpt
