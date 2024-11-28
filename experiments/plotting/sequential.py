import os
import glob
import numpy as np
import pandas as pd

import plotting


def filter_and_group(
    base_dir,
    project_name,
    exp_type,
    algorithm,
    env_name,
    timesteps,
    wrapper_id,
    wrapper_dim,
    wrapper_value,
    seed,
    grouping,
    filtering,
    correct_action_repeat=1
):
    return_dirs = (
        f"eval/returns/{project_name}/exp_{exp_type}/"
        f"{algorithm}__{env_name}__steps_{timesteps}__wid_{wrapper_id}__"
        f"wdim_{wrapper_dim}__wval_{wrapper_value}__seed_{seed}__*/"
    )
    return_dirs = glob.glob(base_dir + return_dirs)

    return_train = []
    return_eval = []

    for return_dir in return_dirs:
        _config = {
            "project_name": return_dir.split("/")[2],
            "exp_type": return_dir.split("/")[3],
            "algorithm": return_dir.split("/")[4].split("__")[0],
            "env_name": return_dir.split("__")[1],
            "timesteps": return_dir.split("__")[2].split("_")[-1],
            "wrapper_id": return_dir.split("__")[3].split("_")[-1],
            "wrapper_dim": return_dir.split("__")[4].split("_")[-1],
            "wrapper_value": return_dir.split("__")[5].split("_")[-1],
            "seed": return_dir.split("__")[6].split("_")[-1]
        }
        _train = dict(np.load(f"{return_dir}/train.npz", allow_pickle=True))
        _eval = dict(np.load(f"{return_dir}/eval.npz", allow_pickle=True))
        train_run_dict = {
            "filter": _config[filtering],
            "group": _config[grouping],
            "seed": _config["seed"],
            "task": "task_all",
            "xs": (_train["timesteps"].flatten() * correct_action_repeat).tolist(),
            "ys": _train["return"].flatten().tolist()
        }
        return_train.append(train_run_dict)

        if "return" in _eval.keys():
            eval_run_dict = {
                "filter": _config[filtering],
                "group": _config[grouping],
                "seed": _config["seed"],
                "task": "task_all",
                "xs": (_eval["timesteps"].flatten() * correct_action_repeat).tolist(),
                "ys": _eval["return"].flatten().tolist()
            }
            return_eval.append(eval_run_dict)
        elif "return_task_0" in _eval.keys():
            for k in _eval.keys():
                if "task" in k:
                    task = f"task_{k.split('_')[-1]}"
                    eval_run_dict = {
                        "filter": _config[filtering],
                        "group": _config[grouping],
                        "seed": _config["seed"],
                        "task": task,
                        "xs": (_eval["timesteps"].flatten() * correct_action_repeat).tolist(),
                        "ys": _eval[k].flatten().tolist()
                    }
                    return_eval.append(eval_run_dict)
        else:
            raise NotImplementedError()

    return return_train, return_eval


def make_plots_basic():
    for algorithm in ["sac", "drq"]:
        for env_name in ["walker-walk-v0", "HalfCheetah-v4"]:
            if env_name == "HalfCheetah-v4" and algorithm == "drq":
                pass
            else:
                for wrapper_dim in ["0", "None"]:

                    base_dir = "../jaxrl3/sequential/"
                    filter_dict = {
                        "project_name": "DynamicsSwitch-sequential-rev",
                        "exp_type": "basic",
                        "algorithm": algorithm,
                        "env_name": env_name,
                        "timesteps": "*",
                        "wrapper_id": "*",
                        "wrapper_dim": wrapper_dim,
                        "wrapper_value": "*",
                        "seed": "*"
                    }
                    return_train, return_eval = filter_and_group(
                        **filter_dict,
                        base_dir=base_dir,
                        grouping="wrapper_value",
                        filtering="wrapper_id",
                        correct_action_repeat=2 if algorithm == "drq" else 1
                    )
                    for returns_type, runs in zip(["train", "eval"], [return_train, return_eval]):
                        save_fname = f"figures/sequential__basic__{env_name}__{algorithm}__wdim_{wrapper_dim}__wid_all__wval_all__{returns_type}"

                        xlim = (0, max(runs[0]["xs"]))
                        bins = np.linspace(*xlim, 30 + 1, endpoint=True)

                        tensor, filters, groups, tasks, seeds = plotting.tensor(runs, bins)

                        fig, axes = plotting.plots(len(filters), cols=4, xticks=2, grid=(2, 2))

                        for filtering, ax in zip(filters, axes):
                            title = filtering
                            ax.set_title(title)
                            ax.set_xlim(*xlim)
                            ax.xaxis.set_major_formatter(plotting.smart_format)

                        for i, filtering in enumerate(filters):
                            ax = axes[i]
                            for j, grouping in enumerate(groups):
                                for k, task in reversed(list(enumerate(tasks))):  # reversed for correct legend
                                    # Aggregate over seeds.
                                    mean = np.nanmean(tensor[i, j, k, :, :], axis=0)
                                    std = np.nanstd(tensor[i, j, k, :, :], axis=0)
                                    plotting.curve(
                                        ax, bins[1:], mean,
                                        low=mean + std / 2,
                                        high=mean - std / 2,
                                        label=grouping,
                                        order=j,
                                        color=plotting.COLORS[j],
                                        linestyle=plotting.LINESTYLES[k]
                                    )
                        # if len(tasks) > 1:
                        #     from matplotlib.lines import Line2D
                        #     custom_legend_element = {
                        #         f"Task {int(task.split('_')[-1]) + 1}": Line2D(
                        #             [0], [0],
                        #             color="k",
                        #             linestyle=plotting.LINESTYLES[k],
                        #         ) for k, task in enumerate(tasks)
                        #     }
                        # else:
                        #     custom_legend_element = None
                        # plotting.legend(fig, adjust=True, custom_elements=custom_legend_element)
                        plotting.legend(fig, adjust=True, ncols=9)

                        # Saves the figure in both PNG and PDF formats and attempts to crop margins off
                        # the PDF.
                        plotting.save(fig, save_fname)


def make_plots_continual():
    for algorithm in ["sac"]:
        for env_name in ["walker-walk-v0", "HalfCheetah-v4"]:
            if env_name == "HalfCheetah-v4" and algorithm == "drq":
                pass
            else:
                for wrapper_dim in ["0", "None"]:

                    base_dir = "../jaxrl3/sequential/"
                    filter_dict = {
                        "project_name": "DynamicsSwitch-sequential-rev",
                        "exp_type": "continual",
                        "algorithm": algorithm,
                        "env_name": env_name,
                        "timesteps": "*",
                        "wrapper_id": "*",
                        "wrapper_dim": wrapper_dim,
                        "wrapper_value": "*",
                        "seed": "*"
                    }
                    return_train, return_eval = filter_and_group(
                        **filter_dict,
                        base_dir=base_dir,
                        grouping="wrapper_value",
                        filtering="wrapper_id",
                        correct_action_repeat=2 if algorithm == "drq" else 1
                    )
                    for returns_type, runs in zip(["train", "eval"], [return_train, return_eval]):
                        save_fname = f"figures/sequential__continual__{env_name}__{algorithm}__wdim_{wrapper_dim}__wid_all__wval_all__{returns_type}"

                        xlim = (0, max(runs[0]["xs"]))
                        bins = np.linspace(*xlim, 300 + 1, endpoint=True)

                        tensor, filters, groups, tasks, seeds = plotting.tensor(runs, bins)

                        # fig, axes = plotting.plots(len(filters), cols=3, size=(2.8, 2.3), xticks=5, grid=(2, 2))
                        fig_size = (2.0, 1.5) if returns_type == "train" else (2.0, 1.7)
                        fig, axes = plotting.plots(len(filters), cols=3, size=fig_size, xticks=5, grid=(2, 2))

                        for filtering, ax in zip(filters, axes):
                            title = filtering
                            ax.set_title(title)
                            ax.set_xlim(*xlim)
                            ax.xaxis.set_major_formatter(plotting.smart_format)

                        for i, filtering in enumerate(filters):
                            ax = axes[i]
                            for j, grouping in enumerate(groups):
                                for k, task in enumerate(tasks):  # reversed for correct legend
                                    # Aggregate over seeds.
                                    mean = np.nanmean(tensor[i, j, k, :, :], axis=0)
                                    std = np.nanstd(tensor[i, j, k, :, :], axis=0)
                                    if returns_type == "eval":
                                        task_label = f"Task {int(task.split('_')[-1]) + 1}"
                                    else:
                                        task_label = None
                                    plotting.curve(
                                        ax, bins[1:], mean,
                                        low=mean + std / 2,
                                        high=mean - std / 2,
                                        label=task_label,
                                        order=j,
                                        color=plotting.COLORS[k] if returns_type == "eval" else "k",
                                        # linestyle=plotting.LINESTYLES[k]
                                    )
                        # if len(tasks) > 1:
                        #     from matplotlib.lines import Line2D
                        #     custom_legend_element = {
                        #         f"Task {int(task.split('_')[-1]) + 1}": Line2D(
                        #             [0], [0],
                        #             color="k",
                        #             linestyle=plotting.LINESTYLES[k],
                        #         ) for k, task in enumerate(tasks)
                        #     }
                        # else:
                        #     custom_legend_element = None
                        # plotting.legend(fig, adjust=True, custom_elements=custom_legend_element)
                        plotting.legend(fig, adjust=True, ncol=5)

                        # Saves the figure in both PNG and PDF formats and attempts to crop margins off
                        # the PDF.
                        plotting.save(fig, save_fname)


def make_plots_basic_selection():
    for algorithm in ["sac"]:
        for env_name in ["walker-walk-v0", "HalfCheetah-v4"]:
            if env_name == "HalfCheetah-v4" and algorithm == "drq":
                pass
            else:
                for wrapper_dim in ["0"]:

                    base_dir = "../jaxrl3/sequential/"
                    filter_dict = {
                        "project_name": "DynamicsSwitch-sequential-rev",
                        "exp_type": "basic",
                        "algorithm": algorithm,
                        "env_name": env_name,
                        "timesteps": "*",
                        "wrapper_id": "*",
                        "wrapper_dim": wrapper_dim,
                        "wrapper_value": "*",
                        "seed": "*"
                    }
                    return_train, return_eval = filter_and_group(
                        **filter_dict,
                        base_dir=base_dir,
                        grouping="wrapper_value",
                        filtering="wrapper_id",
                        correct_action_repeat=2 if algorithm == "drq" else 1
                    )
                    for returns_type, runs in zip(["eval"], [return_eval]):
                        save_fname = f"figures/sequential__basic__selection__{env_name}"

                        xlim = (0, max(runs[0]["xs"]))
                        bins = np.linspace(*xlim, 30 + 1, endpoint=True)

                        tensor, filters, groups, tasks, seeds = plotting.tensor(runs, bins)

                        filter_selection = ["ScaleAction", "InvertAction"]
                        grouping_selection = ["0.2", "0.5", "0.8"]

                        fig, axes = plotting.plots(len(filter_selection), cols=2, xticks=2, grid=(2, 2))

                        for filtering, ax in zip(filter_selection, axes):
                            title = filtering
                            ax.set_title(title)
                            ax.set_xlim(*xlim)
                            ax.xaxis.set_major_formatter(plotting.smart_format)

                        axes_id = 0
                        for i, filtering in reversed(list(enumerate(filters))):
                            if filtering in filter_selection:
                                ax = axes[axes_id]
                                axes_id += 1
                                color_id = 0
                                for j, grouping in enumerate(groups):
                                    if grouping in grouping_selection:
                                        for k, task in reversed(list(enumerate(tasks))):  # reversed for correct legend
                                            # Aggregate over seeds.
                                            mean = np.nanmean(tensor[i, j, k, :, :], axis=0)
                                            std = np.nanstd(tensor[i, j, k, :, :], axis=0)
                                            plotting.curve(
                                                ax, bins[1:], mean,
                                                low=mean + std / 2,
                                                high=mean - std / 2,
                                                label=grouping,
                                                order=j,
                                                color=plotting.COLORS_GRAY[color_id],
                                                linestyle=plotting.LINESTYLES[k]
                                            )
                                        color_id += 1
                        if len(tasks) > 1:
                            from matplotlib.lines import Line2D
                            custom_legend_element = {
                                f"Task {int(task.split('_')[-1]) + 1}": Line2D(
                                    [0], [0],
                                    color="k",
                                    linestyle=plotting.LINESTYLES[k],
                                ) for k, task in enumerate(tasks)
                            }
                        else:
                            custom_legend_element = None
                        plotting.legend(fig, adjust=True, custom_elements=custom_legend_element, ncols=5)
                        # plotting.legend(fig, adjust=True, ncols=9)

                        # Saves the figure in both PNG and PDF formats and attempts to crop margins off
                        # the PDF.
                        plotting.save(fig, save_fname)


def make_plots_basic_selection_poster():
    for algorithm in ["sac"]:
        for env_name in ["walker-walk-v0"]:
            if env_name == "HalfCheetah-v4" and algorithm == "drq":
                pass
            else:
                for wrapper_dim in ["0"]:

                    base_dir = "../jaxrl3/sequential/"
                    filter_dict = {
                        "project_name": "DynamicsSwitch-sequential-rev",
                        "exp_type": "basic",
                        "algorithm": algorithm,
                        "env_name": env_name,
                        "timesteps": "*",
                        "wrapper_id": "*",
                        "wrapper_dim": wrapper_dim,
                        "wrapper_value": "*",
                        "seed": "*"
                    }
                    return_train, return_eval = filter_and_group(
                        **filter_dict,
                        base_dir=base_dir,
                        grouping="wrapper_value",
                        filtering="wrapper_id",
                        correct_action_repeat=2 if algorithm == "drq" else 1
                    )
                    for returns_type, runs in zip(["eval", "train"], [return_eval, return_train]):
                        save_fname = f"figures/sequential__basic__selection__poster__{env_name}__{returns_type}"

                        xlim = (0, max(runs[0]["xs"]))
                        bins = np.linspace(*xlim, 30 + 1, endpoint=True)

                        tensor, filters, groups, tasks, seeds = plotting.tensor(runs, bins)

                        filter_selection = ["InvertAction"]
                        grouping_selection = ["0.5"]

                        fig, axes = plotting.plots(len(filter_selection), cols=1, xticks=2, grid=(1, 1))

                        for filtering, ax in zip(filter_selection, axes):
                            # title = filtering
                            # ax.set_title(title)
                            ax.set_xlim(*xlim)
                            ax.xaxis.set_major_formatter(plotting.smart_format)

                        axes_id = 0
                        for i, filtering in reversed(list(enumerate(filters))):
                            if filtering in filter_selection:
                                ax = axes[axes_id]
                                axes_id += 1
                                color_id = 0
                                for j, grouping in enumerate(groups):
                                    if grouping in grouping_selection:
                                        for k, task in reversed(list(enumerate(tasks))):  # reversed for correct legend
                                            # Aggregate over seeds.
                                            mean = np.nanmean(tensor[i, j, k, :, :], axis=0)
                                            std = np.nanstd(tensor[i, j, k, :, :], axis=0)
                                            plotting.curve(
                                                ax, bins[1:], mean,
                                                low=mean + std / 2,
                                                high=mean - std / 2,
                                                # label=grouping,
                                                order=j,
                                                color=plotting.COLORS_GRAY[color_id],
                                                linestyle=plotting.LINESTYLES[k]
                                            )
                                        color_id += 1
                        if len(tasks) > 1 and returns_type == "eval":
                            from matplotlib.lines import Line2D
                            custom_legend_element = {
                                f"Task {int(task.split('_')[-1]) + 1}": Line2D(
                                    [0], [0],
                                    color="k",
                                    linestyle=plotting.LINESTYLES[k],
                                ) for k, task in enumerate(tasks)
                            }
                        # elif returns_type == "train":
                        #     custom_legend_element = {" ": Line2D(
                        #             [0], [0],
                        #             color="w",
                        #             linestyle=plotting.LINESTYLES[k],
                        #         )
                        #     }
                        else:
                            custom_legend_element = None
                        if custom_legend_element is not None:
                            plotting.legend(
                                fig, adjust=False, custom_elements=custom_legend_element, ncols=1,
                                loc="center right", bbox_to_anchor=(1.4, 0.5),
                                fontsize="medium",
                                # numpoints=1,
                                # labelspacing=0.5,
                                # columnspacing=2.0,
                                handlelength=1.5,
                                # handletextpad=0.8
                            )
                        # plotting.legend(fig, adjust=True, ncols=9)

                        # Saves the figure in both PNG and PDF formats and attempts to crop margins off
                        # the PDF.
                        plotting.save(fig, save_fname)


def make_plots_continual_selection():
    for algorithm in ["sac"]:
        for env_name in ["walker-walk-v0"]:
            if env_name == "HalfCheetah-v4" and algorithm == "drq":
                pass
            else:
                for wrapper_dim in ["0"]:

                    base_dir = "../jaxrl3/sequential/"
                    filter_dict = {
                        "project_name": "DynamicsSwitch-sequential-rev",
                        "exp_type": "continual",
                        "algorithm": algorithm,
                        "env_name": env_name,
                        "timesteps": "*",
                        "wrapper_id": "*",
                        "wrapper_dim": wrapper_dim,
                        "wrapper_value": "*",
                        "seed": "*"
                    }
                    return_train, return_eval = filter_and_group(
                        **filter_dict,
                        base_dir=base_dir,
                        grouping="wrapper_value",
                        filtering="wrapper_id",
                        correct_action_repeat=2 if algorithm == "drq" else 1
                    )
                    for returns_type, runs in zip(["train", "eval"], [return_train, return_eval]):
                        save_fname = f"figures/sequential__continual__selection__{env_name}__{returns_type}"

                        xlim = (0, max(runs[0]["xs"]))
                        bins = np.linspace(*xlim, 300 + 1, endpoint=True)

                        tensor, filters, groups, tasks, seeds = plotting.tensor(runs, bins)

                        # fig, axes = plotting.plots(len(filters), cols=3, size=(2.8, 2.3), xticks=5, grid=(2, 2))
                        fig_size = (2.0, 1.5) if returns_type == "train" else (2.0, 1.7)
                        fig, axes = plotting.plots(len(filters), cols=3, size=fig_size, xticks=5, grid=(2, 2))

                        for filtering, ax in zip(filters, axes):
                            title = filtering
                            ax.set_title(title)
                            ax.set_xlim(*xlim)
                            ax.xaxis.set_major_formatter(plotting.smart_format)

                        for i, filtering in enumerate(filters):
                            ax = axes[i]
                            for j, grouping in enumerate(groups):
                                for k, task in enumerate(tasks):  # reversed for correct legend
                                    # Aggregate over seeds.
                                    mean = np.nanmean(tensor[i, j, k, :, :], axis=0)
                                    std = np.nanstd(tensor[i, j, k, :, :], axis=0)
                                    if returns_type == "eval":
                                        task_label = f"Task {int(task.split('_')[-1]) + 1}"
                                    else:
                                        task_label = None
                                    plotting.curve(
                                        ax, bins[1:], mean,
                                        low=mean + std / 2,
                                        high=mean - std / 2,
                                        label=task_label,
                                        order=j,
                                        color=plotting.COLORS[k] if returns_type == "eval" else "k",
                                        # linestyle=plotting.LINESTYLES[k]
                                    )
                        # if len(tasks) > 1:
                        #     from matplotlib.lines import Line2D
                        #     custom_legend_element = {
                        #         f"Task {int(task.split('_')[-1]) + 1}": Line2D(
                        #             [0], [0],
                        #             color="k",
                        #             linestyle=plotting.LINESTYLES[k],
                        #         ) for k, task in enumerate(tasks)
                        #     }
                        # else:
                        #     custom_legend_element = None
                        # plotting.legend(fig, adjust=True, custom_elements=custom_legend_element)
                        plotting.legend(fig, adjust=True, ncol=5)

                        # Saves the figure in both PNG and PDF formats and attempts to crop margins off
                        # the PDF.
                        plotting.save(fig, save_fname)


def make_plots_continual_slowly():
    for algorithm in ["sac"]:
        for env_name in ["walker-walk-v0", "HalfCheetah-v4"]:
        # for env_name in ["walker-walk-v0"]:
            if env_name == "HalfCheetah-v4" and algorithm == "drq":
                pass
            else:
                for wrapper_dim in ["0", "None"]:

                    base_dir = "../jaxrl3/sequential/"
                    filter_dict = {
                        "project_name": "DynamicsSwitch-sequential-rev",
                        "exp_type": "continual_slowly",
                        "algorithm": algorithm,
                        "env_name": env_name,
                        "timesteps": "*",
                        "wrapper_id": "*",
                        "wrapper_dim": wrapper_dim,
                        "wrapper_value": "*",
                        "seed": "*"
                    }
                    return_train, return_eval = filter_and_group(
                        **filter_dict,
                        base_dir=base_dir,
                        grouping="wrapper_value",
                        filtering="wrapper_id",
                        correct_action_repeat=2 if algorithm == "drq" else 1
                    )
                    for returns_type, runs in zip(["train", "eval"], [return_train, return_eval]):
                        save_fname = f"figures/sequential__continual_slowly__{env_name}__{algorithm}__wdim_{wrapper_dim}__wid_all__wval_all__{returns_type}"

                        xlim = (0, max(runs[0]["xs"]))
                        bins = np.linspace(*xlim, 300 + 1, endpoint=True)

                        tensor, filters, groups, tasks, seeds = plotting.tensor(runs, bins)

                        # fig, axes = plotting.plots(len(filters), cols=3, size=(2.8, 2.3), xticks=5, grid=(2, 2))
                        fig_size = (2.0, 1.5) if returns_type == "train" else (2.0, 1.7)
                        fig, axes = plotting.plots(len(filters), cols=3, size=fig_size, xticks=5, grid=(2, 2))

                        for filtering, ax in zip(filters, axes):
                            title = filtering
                            ax.set_title(title)
                            ax.set_xlim(*xlim)
                            ax.xaxis.set_major_formatter(plotting.smart_format)

                        for i, filtering in enumerate(filters):
                            ax = axes[i]
                            for j, grouping in enumerate(groups):
                                for k, task in enumerate(tasks):  # reversed for correct legend
                                    # Aggregate over seeds.
                                    mean = np.nanmean(tensor[i, j, k, :, :], axis=0)
                                    std = np.nanstd(tensor[i, j, k, :, :], axis=0)
                                    if returns_type == "eval":
                                        task_label = f"Task {int(task.split('_')[-1]) + 1}"
                                    else:
                                        task_label = None
                                    plotting.curve(
                                        ax, bins[1:], mean,
                                        low=mean + std / 2,
                                        high=mean - std / 2,
                                        label=task_label,
                                        order=j,
                                        color=plotting.COLORS[k] if returns_type == "eval" else "k",
                                        # linestyle=plotting.LINESTYLES[k]
                                    )
                        # if len(tasks) > 1:
                        #     from matplotlib.lines import Line2D
                        #     custom_legend_element = {
                        #         f"Task {int(task.split('_')[-1]) + 1}": Line2D(
                        #             [0], [0],
                        #             color="k",
                        #             linestyle=plotting.LINESTYLES[k],
                        #         ) for k, task in enumerate(tasks)
                        #     }
                        # else:
                        #     custom_legend_element = None
                        # plotting.legend(fig, adjust=True, custom_elements=custom_legend_element)
                        plotting.legend(fig, adjust=True, ncol=5)

                        # Saves the figure in both PNG and PDF formats and attempts to crop margins off
                        # the PDF.
                        plotting.save(fig, save_fname)


def make_plots_parallel():
    for algorithm in ["sac"]:
        for env_name in ["walker-walk-v0"]:
            if env_name == "HalfCheetah-v4" and algorithm == "drq":
                pass
            else:
                for wrapper_dim in ["0", "None"]:
                    for exp_type in ["parallel_offset", "parallel_scale"]:

                        base_dir = "../jaxrl3/sequential/"
                        filter_dict = {
                            "project_name": "DynamicsSwitch-sequential-rev",
                            "exp_type": exp_type,
                            "algorithm": algorithm,
                            "env_name": env_name,
                            "timesteps": "*",
                            "wrapper_id": "*",
                            "wrapper_dim": wrapper_dim,
                            "wrapper_value": "*",
                            "seed": "*"
                        }
                        return_train, return_eval = filter_and_group(
                            **filter_dict,
                            base_dir=base_dir,
                            grouping="wrapper_value",
                            filtering="wrapper_id",
                            correct_action_repeat=2 if algorithm == "drq" else 1
                        )
                        for returns_type, runs in zip(["train", "eval"], [return_train, return_eval]):
                            save_fname = f"figures/sequential__{exp_type}__{env_name}__{algorithm}__wdim_{wrapper_dim}__wid_all__wval_all__{returns_type}"

                            xlim = (0, max(runs[0]["xs"]))
                            bins = np.linspace(*xlim, 30 + 1, endpoint=True)

                            tensor, filters, groups, tasks, seeds = plotting.tensor(runs, bins)

                            fig, axes = plotting.plots(len(filters), cols=4, xticks=2, grid=(2, 2))

                            for filtering, ax in zip(filters, axes):
                                title = filtering
                                ax.set_title(title)
                                ax.set_xlim(*xlim)
                                ax.xaxis.set_major_formatter(plotting.smart_format)

                            for i, filtering in enumerate(filters):
                                ax = axes[i]
                                for j, grouping in enumerate(groups):
                                    for k, task in reversed(list(enumerate(tasks))):  # reversed for correct legend
                                        # Aggregate over seeds.
                                        mean = np.nanmean(tensor[i, j, k, :, :], axis=0)
                                        std = np.nanstd(tensor[i, j, k, :, :], axis=0)
                                        plotting.curve(
                                            ax, bins[1:], mean,
                                            low=mean + std / 2,
                                            high=mean - std / 2,
                                            label=grouping,
                                            order=j,
                                            color=plotting.COLORS[j],
                                            linestyle=plotting.LINESTYLES[k]
                                        )
                            # if len(tasks) > 1:
                            #     from matplotlib.lines import Line2D
                            #     custom_legend_element = {
                            #         f"Task {int(task.split('_')[-1]) + 1}": Line2D(
                            #             [0], [0],
                            #             color="k",
                            #             linestyle=plotting.LINESTYLES[k],
                            #         ) for k, task in enumerate(tasks)
                            #     }
                            # else:
                            #     custom_legend_element = None
                            # plotting.legend(fig, adjust=True, custom_elements=custom_legend_element)
                            plotting.legend(fig, adjust=True, ncols=9)

                            # Saves the figure in both PNG and PDF formats and attempts to crop margins off
                            # the PDF.
                            plotting.save(fig, save_fname)


def compute_seq_avg_performance(exp_type):
    infos = []
    for algorithm in ["sac", "drq"]:
        if exp_type == "continual" and algorithm == "drq":
            pass
        else:
            for env_name in ["walker-walk-v0", "HalfCheetah-v4"]:
                if env_name == "HalfCheetah-v4" and algorithm == "drq":
                    pass
                else:
                    for wrapper_dim in ["0", "None"]:

                        base_dir = "../jaxrl3/sequential/"
                        filter_dict = {
                            "project_name": "DynamicsSwitch-sequential-rev",
                            "exp_type": exp_type,
                            "algorithm": algorithm,
                            "env_name": env_name,
                            "timesteps": "*",
                            "wrapper_id": "*",
                            "wrapper_dim": wrapper_dim,
                            "wrapper_value": "*",
                            "seed": "*"
                        }
                        return_train, return_eval = filter_and_group(
                            **filter_dict,
                            base_dir=base_dir,
                            grouping="wrapper_value",
                            filtering="wrapper_id",
                            correct_action_repeat=2 if algorithm == "drq" else 1
                        )

                        if exp_type in ["basic"]:
                            ats = np.linspace(0, len(return_eval[0]["xs"]) - 1, 2 + 1, endpoint=True, dtype=int).tolist()
                        elif exp_type in ["continual"]:
                            ats = np.linspace(0, len(return_eval[0]["xs"]) - 1, 10 + 1, endpoint=True, dtype=int).tolist()
                        else:
                            raise NotImplementedError()
                        tensor, filters, groups, tasks, seeds = plotting.tensor_at(return_eval, ats)

                        if exp_type in ["basic"]:
                            assert len(tasks) == 2
                        elif exp_type in ["continual"]:
                            assert len(tasks) == 10
                        else:
                            raise NotImplementedError()

                        for i, filtering in enumerate(filters):
                            for j, grouping in enumerate(groups):
                                max_over_tasks = np.max(np.nanmean(tensor[i, j, :, :, :], axis=1))

                                mean_over_tasks = np.nanmean(tensor[i, j, :, :, :] / max_over_tasks, axis=0)
                                # aggregate seeds
                                mean = np.nanmean(mean_over_tasks, axis=0)
                                std = np.nanstd(mean_over_tasks, axis=0)
                                info = {}
                                info["algorithm"] = algorithm
                                info["env_name"] = env_name
                                info["wrapper_dim"] = wrapper_dim
                                info["filter"] = filtering
                                info["group"] = grouping
                                info["mean"] = np.around(mean, 3)
                                info["std"] = np.around(std / 2, 3)
                                infos.append(info)
    return infos


def compute_seq_forward_transfer(exp_type):
    infos = []
    for algorithm in ["sac", "drq"]:
        if exp_type == "continual" and algorithm == "drq":
            pass
        else:
            for env_name in ["walker-walk-v0", "HalfCheetah-v4"]:
                if env_name == "HalfCheetah-v4" and algorithm == "drq":
                    pass
                else:
                    for wrapper_dim in ["0", "None"]:

                        base_dir = "../jaxrl3/sequential/"
                        filter_dict = {
                            "project_name": "DynamicsSwitch-sequential-rev",
                            "exp_type": exp_type,
                            "algorithm": algorithm,
                            "env_name": env_name,
                            "timesteps": "*",
                            "wrapper_id": "*",
                            "wrapper_dim": wrapper_dim,
                            "wrapper_value": "*",
                            "seed": "*"
                        }
                        return_train, return_eval = filter_and_group(
                            **filter_dict,
                            base_dir=base_dir,
                            grouping="wrapper_value",
                            filtering="wrapper_id",
                            correct_action_repeat=2 if algorithm == "drq" else 1
                        )

                        if exp_type in ["basic"]:
                            ats = np.linspace(0, len(return_eval[0]["xs"]) - 1, 2 + 1, endpoint=True, dtype=int).tolist()
                        elif exp_type in ["continual"]:
                            ats = np.linspace(0, len(return_eval[0]["xs"]) - 1, 10 + 1, endpoint=True, dtype=int).tolist()
                        else:
                            raise NotImplementedError()
                        tensor, filters, groups, tasks, seeds = plotting.tensor_at(return_eval, ats)

                        if exp_type in ["basic"]:
                            assert len(tasks) == 2
                        elif exp_type in ["continual"]:
                            assert len(tasks) == 10
                        else:
                            raise NotImplementedError()
                        for i, filtering in enumerate(filters):
                            for j, grouping in enumerate(groups):
                                max_over_tasks = np.max(np.nanmean(tensor[i, j, :, :, :], axis=1))
                                # gather FTj
                                ftjs = []
                                for j_task in range(1, len(tasks)):
                                    for i_task in range(1, j_task + 1):
                                        ftjs.append((tensor[i, j, j_task, :, i_task] - tensor[i, j, j_task, :, i_task - 1]) / max_over_tasks)

                                mean_over_ftjs = np.nanmean(ftjs, axis=0)
                                # aggregate seeds
                                mean = np.nanmean(mean_over_ftjs, axis=0)
                                std = np.nanstd(mean_over_ftjs, axis=0)
                                info = {}
                                info["algorithm"] = algorithm
                                info["env_name"] = env_name
                                info["wrapper_dim"] = wrapper_dim
                                info["filter"] = filtering
                                info["group"] = grouping
                                info["mean"] = np.around(mean, 3)
                                info["std"] = np.around(std / 2, 3)
                                infos.append(info)
    return infos


def compute_seq_backward_transfer(exp_type):
    infos = []
    for algorithm in ["sac", "drq"]:
        if exp_type == "continual" and algorithm == "drq":
            pass
        else:
            for env_name in ["walker-walk-v0", "HalfCheetah-v4"]:
                if env_name == "HalfCheetah-v4" and algorithm == "drq":
                    pass
                else:
                    for wrapper_dim in ["0", "None"]:

                        base_dir = "../jaxrl3/sequential/"
                        filter_dict = {
                            "project_name": "DynamicsSwitch-sequential-rev",
                            "exp_type": exp_type,
                            "algorithm": algorithm,
                            "env_name": env_name,
                            "timesteps": "*",
                            "wrapper_id": "*",
                            "wrapper_dim": wrapper_dim,
                            "wrapper_value": "*",
                            "seed": "*"
                        }
                        return_train, return_eval = filter_and_group(
                            **filter_dict,
                            base_dir=base_dir,
                            grouping="wrapper_value",
                            filtering="wrapper_id",
                            correct_action_repeat=2 if algorithm == "drq" else 1
                        )

                        if exp_type in ["basic"]:
                            ats = np.linspace(0, len(return_eval[0]["xs"]) - 1, 2 + 1, endpoint=True, dtype=int).tolist()
                        elif exp_type in ["continual"]:
                            ats = np.linspace(0, len(return_eval[0]["xs"]) - 1, 10 + 1, endpoint=True, dtype=int).tolist()
                        else:
                            raise NotImplementedError()
                        tensor, filters, groups, tasks, seeds = plotting.tensor_at(return_eval, ats)

                        if exp_type in ["basic"]:
                            assert len(tasks) == 2
                        elif exp_type in ["continual"]:
                            assert len(tasks) == 10
                        else:
                            raise NotImplementedError()
                        for i, filtering in enumerate(filters):
                            for j, grouping in enumerate(groups):
                                max_over_tasks = np.max(np.nanmean(tensor[i, j, :, :, :], axis=1))
                                # gather BTi
                                btis = []
                                for i_task in range(0, len(tasks) - 1):
                                    for j_task in range(i_task + 2, len(tasks) + 1):
                                        bti = (tensor[i, j, i_task, :, j_task] - tensor[i, j, i_task, :, j_task - 1]) / max_over_tasks
                                        btis.append(np.max([np.zeros_like(bti), bti], axis=0))

                                mean_over_btis = np.nanmean(btis, axis=0)
                                # aggregate seeds
                                mean = np.nanmean(mean_over_btis, axis=0)
                                std = np.nanstd(mean_over_btis, axis=0)
                                info = {}
                                info["algorithm"] = algorithm
                                info["env_name"] = env_name
                                info["wrapper_dim"] = wrapper_dim
                                info["filter"] = filtering
                                info["group"] = grouping
                                info["mean"] = np.around(mean, 3)
                                info["std"] = np.around(std / 2, 3)
                                infos.append(info)
    return infos


def compute_seq_forgetting(exp_type):
    infos = []
    for algorithm in ["sac", "drq"]:
        if exp_type == "continual" and algorithm == "drq":
            pass
        else:
            for env_name in ["walker-walk-v0", "HalfCheetah-v4"]:
                if env_name == "HalfCheetah-v4" and algorithm == "drq":
                    pass
                else:
                    for wrapper_dim in ["0", "None"]:

                        base_dir = "../jaxrl3/sequential/"
                        filter_dict = {
                            "project_name": "DynamicsSwitch-sequential-rev",
                            "exp_type": exp_type,
                            "algorithm": algorithm,
                            "env_name": env_name,
                            "timesteps": "*",
                            "wrapper_id": "*",
                            "wrapper_dim": wrapper_dim,
                            "wrapper_value": "*",
                            "seed": "*"
                        }
                        return_train, return_eval = filter_and_group(
                            **filter_dict,
                            base_dir=base_dir,
                            grouping="wrapper_value",
                            filtering="wrapper_id",
                            correct_action_repeat=2 if algorithm == "drq" else 1
                        )

                        if exp_type in ["basic"]:
                            ats = np.linspace(0, len(return_eval[0]["xs"]) - 1, 2 + 1, endpoint=True, dtype=int).tolist()
                        elif exp_type in ["continual"]:
                            ats = np.linspace(0, len(return_eval[0]["xs"]) - 1, 10 + 1, endpoint=True, dtype=int).tolist()
                        else:
                            raise NotImplementedError()
                        tensor, filters, groups, tasks, seeds = plotting.tensor_at(return_eval, ats)

                        if exp_type in ["basic"]:
                            assert len(tasks) == 2
                        elif exp_type in ["continual"]:
                            assert len(tasks) == 10
                        else:
                            raise NotImplementedError()
                        for i, filtering in enumerate(filters):
                            for j, grouping in enumerate(groups):
                                max_over_tasks = np.max(np.nanmean(tensor[i, j, :, :, :], axis=1))
                                # gather Fi
                                fis = []
                                for i_task in range(0, len(tasks) - 1):
                                    for j_task in range(i_task + 2, len(tasks) + 1):
                                        fi = (tensor[i, j, i_task, :, j_task - 1] - tensor[i, j, i_task, :, j_task]) / max_over_tasks
                                        fis.append(np.max([np.zeros_like(fi), fi], axis=0))

                                mean_over_fis = np.nanmean(fis, axis=0)
                                # aggregate seeds
                                mean = np.nanmean(mean_over_fis, axis=0)
                                std = np.nanstd(mean_over_fis, axis=0)
                                info = {}
                                info["algorithm"] = algorithm
                                info["env_name"] = env_name
                                info["wrapper_dim"] = wrapper_dim
                                info["filter"] = filtering
                                info["group"] = grouping
                                info["mean"] = np.around(mean, 3)
                                info["std"] = np.around(std / 2, 3)
                                infos.append(info)
    return infos


def make_tables():
    for exp_type in ["basic", "continual"]:
        infos_ap = compute_seq_avg_performance(exp_type)
        infos_ft = compute_seq_forward_transfer(exp_type)
        infos_bt = compute_seq_backward_transfer(exp_type)
        infos_f = compute_seq_forgetting(exp_type)

        # keep last value for avg performance
        for i, info in enumerate(infos_ap):
            infos_ap[i]["mean"] = infos_ap[i]["mean"][-1]
            infos_ap[i]["std"] = infos_ap[i]["std"][-1]

        for algorithm in ["sac", "drq"]:
            for env_name in ["walker-walk-v0", "HalfCheetah-v4"]:
                if algorithm == "drq" and env_name == "HalfCheetah-v4":
                    continue
                if algorithm == "drq" and exp_type == "continual":
                    continue
                dfs = []
                for metric_name, metric_infos in zip(["Avg. performance", "Fwd. transfer", "Bwd. transfer", "Forgetting"], [infos_ap, infos_ft, infos_bt, infos_f]):
                    df = pd.DataFrame(metric_infos)
                    df = df.loc[df["algorithm"] == algorithm]
                    df = df.loc[df["env_name"] == env_name]
                    df = df.round(2)
                    df = df.loc[~((df["filter"].isin(["InvertAction", "SwapAction"])) & (df["group"].isin(["0.1", "0.2", "0.3", "0.4", "0.6", "0.7", "0.8", "0.9"])))]
                    df = df.loc[df["group"].isin(["0.2", "0.5", "0.8"])]
                    df = df.reset_index(drop=True)
                    df.loc[df["filter"].isin(["InvertAction", "SwapAction"]), "group"] = "-"
                    df.loc[df["wrapper_dim"] == "None", "wrapper_dim"] = "all"
                    df[metric_name] = df["mean"].astype(str) + " (" + df["std"].astype(str) + ")"
                    df = df.drop(columns=["mean", "std", "algorithm", "env_name"])
                    if exp_type == "basic":
                        df = df.set_index(["filter", "wrapper_dim", "group"])
                        df.index.names = ["Wrapper", "Dim.", "Value"]
                    elif exp_type == "continual":
                        df = df.drop(columns="group")
                        df = df.set_index(["filter", "wrapper_dim"])
                        df.index.names = ["Wrapper", "Dim."]
                    df = df.sort_index(level="Wrapper")
                    dfs.append(df)

                df = pd.concat(dfs, axis=1)

                base_dir = "tables/"
                os.makedirs(base_dir, exist_ok=True)
                fname = base_dir + f"sequential__{exp_type}__{env_name}__{algorithm}.tex"
                df.to_latex(fname, multirow=True)
                # custom modifications
                with open(fname, 'r') as file:
                    # read a list of lines into data
                    data = file.readlines()

                if exp_type == "basic":
                    data[0] = '\\begin{tabular}{lllllll}\n'
                    data[2] = '\\multicolumn{1}{c}{\\bf Wrapper}  &\\multicolumn{1}{c}{\\bf Dim.}  &\\multicolumn{1}{c}{\\bf Value}  &\\multicolumn{1}{c}{\\bf Avg. performance}  &\\multicolumn{1}{c}{\\bf Fwd. transfer} &\\multicolumn{1}{c}{\\bf Bwd. transfer} &\\multicolumn{1}{c}{\\bf Forgetting}\n'
                elif exp_type == "continual":
                    data[0] = '\\begin{tabular}{llllll}\n'
                    data[2] = '\\multicolumn{1}{c}{\\bf Wrapper}  &\\multicolumn{1}{c}{\\bf Dim.}  &\\multicolumn{1}{c}{\\bf Avg. performance}  &\\multicolumn{1}{c}{\\bf Fwd. transfer} &\\multicolumn{1}{c}{\\bf Bwd. transfer} &\\multicolumn{1}{c}{\\bf Forgetting}\n'
                data[3] = '\\\\ \\hline \\\\\n'
                data.insert(0, '{\\renewcommand{\\arraystretch}{1.2}%\n')
                data.append('}\n')
                data.pop(-3)
                data.pop(5)
                data.pop(2)

                # and write everything back
                with open(fname, 'w') as file:
                    file.writelines(data)


if __name__ == "__main__":
    # make_plots_basic()
    # make_plots_continual()
    # make_plots_basic_selection()
    # make_plots_continual_selection()
    # make_plots_continual_slowly()
    # make_plots_parallel()
    # make_tables()
    make_plots_basic_selection_poster()

    # infos = compute_seq_avg_performance("basic")
    # infos = compute_seq_avg_performance("continual")
    # infos = compute_seq_forward_transfer("basic")
    # infos = compute_seq_forward_transfer("continual")
    # infos = compute_seq_backward_transfer("basic")
    # infos = compute_seq_backward_transfer("continual")
    # infos = compute_seq_forgetting("basic")
    # infos = compute_seq_forgetting("continual")
