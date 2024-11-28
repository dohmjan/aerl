from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

extras = {
    "mujoco": ["mujoco<3.0,>=2.3.3"],
    "dm-control": ["shimmy==1.3.0", "dm-control>=1.0.10,<=1.0.14", "imageio", "h5py>=3.7.0"],
    "robotics": ["gymnasium_robotics==1.2.4"],
    "sb3": ["stable-baselines3==2.2.1"],
    "jaxrl": ["jaxrl3 @ git+https://github.com/dohmjan/jaxrl3.git@main"]
}
extras["all"] = [
    lib for key, libs in extras.items() for lib in libs
]
extras["testing"] = [lib for key, libs in extras.items() if key != "sb3" for lib in libs]
extras["testing"].append("pytest")

setup(
    name='aerl',
    version='0.0.1',
    packages=['aerl'],
    desription='Action effect suite for reinforcement learning',
    license='MIT',
    install_requires=["gymnasium==0.29.1"],
    tests_require=extras["testing"],
    extras_require=extras,
)

