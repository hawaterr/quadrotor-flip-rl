"""Asymmetric actor-critic PPO policy: the actor sees only the first n_actor observations."""
from torch import Tensor, nn
from stable_baselines3.common.policies import ActorCriticPolicy


def mlp(n_in: int, sizes: list[int], activation_fn: type[nn.Module]) -> tuple[nn.Sequential, int]:
    """The layers and their output size."""
    layers = []
    for size in sizes:
        layers += [nn.Linear(n_in, size), activation_fn()]
        n_in = size
    return nn.Sequential(*layers), n_in


class SplitMlpExtractor(nn.Module):
    """Same interface as SB3's MlpExtractor; the policy net sees only the actor part of the observation."""

    def __init__(self, n_actor: int, n_obs: int, net_arch: list[int] | dict[str, list[int]],
                 activation_fn: type[nn.Module]):
        super().__init__()
        pi_arch, vf_arch = (net_arch["pi"], net_arch["vf"]) if isinstance(net_arch, dict) else (net_arch, net_arch)
        self.n_actor = n_actor
        self.policy_net, self.latent_dim_pi = mlp(n_actor, pi_arch, activation_fn)
        self.value_net, self.latent_dim_vf = mlp(n_obs, vf_arch, activation_fn)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: Tensor) -> Tensor:
        return self.policy_net(features[..., :self.n_actor])

    def forward_critic(self, features: Tensor) -> Tensor:
        return self.value_net(features)


class AsymmetricPolicy(ActorCriticPolicy):
    """ActorCriticPolicy whose actor sees only obs[:n_actor] (pass n_actor in policy_kwargs)."""

    def __init__(self, *args, n_actor: int, **kwargs):
        self.n_actor = n_actor  # set before super().__init__, which builds the networks
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = SplitMlpExtractor(self.n_actor, self.features_dim, self.net_arch, self.activation_fn)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), "n_actor": self.n_actor}
