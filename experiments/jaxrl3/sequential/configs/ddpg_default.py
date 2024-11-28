import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4

    config.gamma = 0.99
    config.tau = 0.005

    config.policy_frequency = 2

    config.exploration_noise = 0.1

    return config
