import json
import matplotlib.pyplot as plt
import pathlib
import os

import plotting

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Roman",
    "axes.titlesize": "medium",
    "font.size": 8.0,
})


def load_logged_json(logdir: str, filename: str):
    # logdir = elements.path.Path(logdir)

    data = []
    with open(logdir + filename, "r") as file:
        for row in file:
            data.append(json.loads(row))
    return data


def make_plot_fetch_trajectories():
    xy_trajectories = load_logged_json("../relearning/", "FetchReach_InvertAction_trajectories.json")

    if len(xy_trajectories) == 0:
        return

    # fig, ax = plt.subplots(figsize=(6.5, 2))
    fig, ax = plt.subplots(figsize=(4.875, 1.5))
    # num_trajectories = len(xy_trajectories)
    for i, traj in enumerate(xy_trajectories):
        if i in [19]:
            ax.plot(traj["agent_x"], traj["agent_y"], color="k", alpha=1.0, ls="--")
        if i in [20, 25, 30, 35, 36]:
            ax.plot(traj["agent_x"], traj["agent_y"], color="k", alpha=1.0)
        # ax.plot(traj["agent_x"], traj["agent_y"], color="k", alpha=(i / num_trajectories))
    if "block_x" in xy_trajectories[0].keys():
        ax.scatter(xy_trajectories[0]["block_x"], xy_trajectories[0]["block_y"], color="blue", marker="s", label="Block")
    ax.scatter(xy_trajectories[0]["goal_x"], xy_trajectories[0]["goal_y"], color="g", marker="*", s=100, label="Goal")
    ax.scatter(xy_trajectories[0]["start_x"], xy_trajectories[0]["start_y"], color="g", marker="o", s=64, label="Start")

    circle = plt.Circle((xy_trajectories[0]["goal_x"], xy_trajectories[0]["goal_y"]), 0.05, color='g', fill=True, alpha=0.1)
    ax.add_patch(circle)

    ax.set_xlim(1.15, 1.8)
    ax.set_ylim(0.58, 0.78)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")

    ax.text(1.232, 0.682, "0", color="0.4")

    ax.text(1.450, 0.642, "1", color="0.4")
    ax.text(1.382, 0.630, "2", color="0.4")
    ax.text(1.305, 0.628, "3", color="0.4")
    ax.text(1.260, 0.635, "4", color="0.4")
    ax.text(1.242, 0.642, "5", color="0.4")

    ax.annotate(
        "", xy=(1.48, 0.635), xytext=(1.22, 0.635),
        arrowprops=dict(arrowstyle="<-", color="0.4", connectionstyle="arc3,rad=0.2"),
    )
    ax.text(1.32, 0.59, "relearn", color="0.4")

    leg = ax.legend(loc="upper right")
    leg.get_frame().set_edgecolor('white')

    save_fname = "figures/relearning__basic__FetchReach-v2__wdim_0_wid_ActionInvert__eval"

    plotting.save(fig, save_fname)


if __name__ == "__main__":
    make_plot_fetch_trajectories()
