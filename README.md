# Action effect suite in reinforcement learning
Enabling reinforcement learning (RL) agents to adapt to or be robust against changing environment dynamics is crucial for many applications. To facilitate systematic research in that regard, we contribute a universal _action effect suite_. 
It perturbs the action effect of any given environment, regardless of input modalities, in a multitude of ways. This can be leveraged to thoroughly analyze and enhance robustness and transfer capabilities of RL agents.

## Installation
The _action effect suite_ currently supports 'python>=3.8' and is compliant with the API standard for reinforcement learning introduced in 'gymnasium>=0.26.0,<=0.29.1' (c.f. [gymnasium docs](https://gymnasium.farama.org/introduction/migration_guide/)). It can be installed directly from the repository:
```sh
git clone https://github.com/dohmjan/aerl.git
cd aerl/
pip install -e .
# pip install -e .[mujoco]      # for gymnasium mujoco environments
# pip install -e .[dm-control]  # for dm-control environments via shimmy
# pip install -e .[robotics]    # for gymnasium robotics environments
# pip install -e .[sb3]         # for examples using StableBaselines 3
# pip install -e .[all]         # for everything of the above
# pip install -e .[testing]     # for tests

```


## Usage
Action effect modifications are just wrapped around the environment instance.
```python
import gymnasium as gym
import numpy as np
from aerl import InvertAction

env = gym.make("Hopper-v4")
env = InvertAction(env, dim=0, toggle_at_step=0)
_ = env.reset(seed=2)
intended_action = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_ = env.step(intended_action)
# Executes the modified action np.array([-0.5, 0.5, 0.5]) in the base environment.
```
More examples are provided [here](examples/sb3/) and in the [docstrings](aerl/wrappers.py) of the corresponding wrapper.

