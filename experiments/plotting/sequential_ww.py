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
    return_dirs2 = (
        f"eval/returns/{project_name}/exp_{exp_type}/"
        f"{algorithm}__{env_name}__steps_{timesteps}__wid_{wrapper_id}__"
        f"wdim_{wrapper_dim}__wval_{wrapper_value}__wrep_*__seed_{seed}__*/"
    )
    return_dirs = glob.glob(base_dir + return_dirs)
    return_dirs2 = glob.glob(base_dir + return_dirs2)
    return_dirs = return_dirs + return_dirs2

    return_train = []
    return_eval = []

    for return_dir in return_dirs:
        _config = {
            "project_name": return_dir.split("/")[2],
            "exp_type": return_dir.split("/")[3],
            "algorithm": return_dir.split("/")[4].split("__")[0],
            "env_name": return_dir.split("__")[1],
            "timesteps": return_dir.split("steps_")[-1].split("__")[0],
            "wrapper_id": return_dir.split("wid_")[-1].split("__")[0],
            "wrapper_dim": return_dir.split("wdim_")[-1].split("__")[0],
            "wrapper_value": return_dir.split("wval_")[-1].split("__")[0],
            # "wrapper_repeat": return_dir.split("wrep_")[-1].split("__")[0],
            "seed": return_dir.split("seed_")[-1].split("__")[0]
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


def make_plots_basic_selection():
    for algorithm in ["sac"]:
        for env_name in ["walker-walk-v0"]:
            if env_name == "HalfCheetah-v4" and algorithm == "drq":
                pass
            else:
                for wrapper_dim in ["0"]:
                    ts = [2000000] if algorithm == "drq" else [1000000]
                    for timesteps in ts:

                        base_dir = "../jaxrl3/sequential/"
                        filter_dict = {
                            "project_name": "DynamicsSwitch-sequential-2024-0",
                            "exp_type": "basic",
                            "algorithm": algorithm,
                            "env_name": env_name,
                            "timesteps": timesteps,
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
                                                    # ax, bins[1:], mean,
                                                    ax, bins, mean,
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
                                    f"CTX {int(task.split('_')[-1]) + 1}": Line2D(
                                        [0], [0],
                                        color="k",
                                        linestyle=plotting.LINESTYLES[k],
                                    ) for k, task in enumerate(tasks)
                                }
                            else:
                                custom_legend_element = None
                            plotting.legend(
                                fig,
                                adjust=True,
                                custom_elements=custom_legend_element,
                                ncol=5,
                                plotpad=0.5,
                                title="scaling factors",
                                legendpad=(0, -0.05)
                            )
                            # plotting.legend(fig, adjust=True, ncols=9)

                            # Saves the figure in both PNG and PDF formats and attempts to crop margins off
                            # the PDF.
                            plotting.save(fig, save_fname)


def compute_seq_avg_performance(exp_type="basic"):
    infos = []
    for algorithm in ["sac", "ddpg", "drq"]:
        if exp_type == "continual" and algorithm == "drq":
            pass
        else:
            for env_name in ["walker-walk-v0"]:
                if env_name == "HalfCheetah-v4" and algorithm == "drq":
                    pass
                else:
                    for wrapper_dim in ["0"]:
                        for wrapper_id in ["ScaleAction"]:
                            ts = 2000000 if algorithm == "drq" else 1000000
                            for timesteps in [ts]:
                                base_dir = "../jaxrl3/sequential/"
                                filter_dict = {
                                    "project_name": "DynamicsSwitch-sequential-2024-0",
                                    "exp_type": exp_type,
                                    "algorithm": algorithm,
                                    "env_name": env_name,
                                    "timesteps": timesteps,
                                    "wrapper_id": wrapper_id,
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
                                        for seed, ys in enumerate(mean_over_tasks):
                                            info = {}
                                            info["filter"] = "Avg. Performance"
                                            info["group"] = algorithm
                                            info["seed"] = seed
                                            info["task"] = "dummy"
                                            info["xs"] = float(grouping)
                                            info["ys"] = ys[-1]
                                            infos.append(info)
    return infos


def compute_seq_forward_transfer(exp_type="basic"):
    infos = []
    for algorithm in ["sac", "ddpg", "drq"]:
        if exp_type == "continual" and algorithm == "drq":
            pass
        else:
            for env_name in ["walker-walk-v0"]:
                if env_name == "HalfCheetah-v4" and algorithm == "drq":
                    pass
                else:
                    for wrapper_dim in ["0"]:
                        for wrapper_id in ["ScaleAction"]:
                            ts = 2000000 if algorithm == "drq" else 1000000
                            for timesteps in [ts]:
                                base_dir = "../jaxrl3/sequential/"
                                filter_dict = {
                                    "project_name": "DynamicsSwitch-sequential-2024-0",
                                    "exp_type": exp_type,
                                    "algorithm": algorithm,
                                    "env_name": env_name,
                                    "timesteps": timesteps,
                                    "wrapper_id": wrapper_id,
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
                                        for seed, ys in enumerate(mean_over_ftjs):
                                            info = {}
                                            info["filter"] = "Forward Transfer"
                                            info["group"] = algorithm
                                            info["seed"] = seed
                                            info["task"] = "dummy"
                                            info["xs"] = float(grouping)
                                            info["ys"] = ys
                                            infos.append(info)
    return infos


def compute_seq_backward_transfer(exp_type="basic"):
    infos = []
    for algorithm in ["sac", "ddpg", "drq"]:
        if exp_type == "continual" and algorithm == "drq":
            pass
        else:
            for env_name in ["walker-walk-v0"]:
                if env_name == "HalfCheetah-v4" and algorithm == "drq":
                    pass
                else:
                    for wrapper_dim in ["0"]:
                        for wrapper_id in ["ScaleAction"]:
                            ts = 2000000 if algorithm == "drq" else 1000000
                            for timesteps in [ts]:
                                base_dir = "../jaxrl3/sequential/"
                                filter_dict = {
                                    "project_name": "DynamicsSwitch-sequential-2024-0",
                                    "exp_type": exp_type,
                                    "algorithm": algorithm,
                                    "env_name": env_name,
                                    "timesteps": timesteps,
                                    "wrapper_id": wrapper_id,
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
                                        for seed, ys in enumerate(mean_over_btis):
                                            info = {}
                                            info["filter"] = "Backward Transfer"
                                            info["group"] = algorithm
                                            info["seed"] = seed
                                            info["task"] = "dummy"
                                            info["xs"] = float(grouping)
                                            info["ys"] = ys
                                            infos.append(info)
    return infos


def compute_seq_forgetting(exp_type="basic"):
    infos = []
    for algorithm in ["sac", "ddpg", "drq"]:
        if exp_type == "continual" and algorithm == "drq":
            pass
        else:
            for env_name in ["walker-walk-v0"]:
                if env_name == "HalfCheetah-v4" and algorithm == "drq":
                    pass
                else:
                    for wrapper_dim in ["0"]:
                        for wrapper_id in ["ScaleAction"]:
                            ts = 2000000 if algorithm == "drq" else 1000000
                            for timesteps in [ts]:
                                base_dir = "../jaxrl3/sequential/"
                                filter_dict = {
                                    "project_name": "DynamicsSwitch-sequential-2024-0",
                                    "exp_type": exp_type,
                                    "algorithm": algorithm,
                                    "env_name": env_name,
                                    "timesteps": timesteps,
                                    "wrapper_id": wrapper_id,
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
                                        for seed, ys in enumerate(mean_over_fis):
                                            info = {}
                                            info["filter"] = "Forgetting"
                                            info["group"] = algorithm
                                            info["seed"] = seed
                                            info["task"] = "dummy"
                                            info["xs"] = float(grouping)
                                            info["ys"] = ys
                                            infos.append(info)
    return infos


def aggregate_over_groups(infos, num_groups=21, seeds=5):
    infos_agg = []
    for algorithm in ["sac", "ddpg", "drq"]:
        for seed in range(seeds):
            info_agg = {}
            info_agg["filter"] = infos[0]["filter"]
            info_agg["group"] = algorithm.upper()
            info_agg["task"] = infos[0]["task"]
            info_agg["seed"] = seed
            info_agg["xs"] = []
            info_agg["ys"] = []
            for info in infos:
                if (info["seed"] == seed) and (info["group"] == algorithm):
                    info_agg["xs"].append(info["xs"])
                    info_agg["ys"].append(info["ys"])
            # sort
            sort_idx = np.argsort(info_agg["xs"])
            info_agg["xs"] = np.array(info_agg["xs"])[sort_idx].tolist()
            info_agg["ys"] = np.array(info_agg["ys"])[sort_idx].tolist()
            infos_agg.append(info_agg)
    # assert len(infos_agg[0]["xs"]) == num_groups
    return infos_agg


def make_plots_transfer():
    infos_ap = compute_seq_avg_performance()
    infos_ap = aggregate_over_groups(infos_ap)

    infos_ft = compute_seq_forward_transfer()
    infos_ft = aggregate_over_groups(infos_ft)

    # infos_bt = compute_seq_backward_transfer()
    # infos_bt = aggregate_over_groups(infos_bt)

    infos_f = compute_seq_forgetting()
    infos_f = aggregate_over_groups(infos_f)

    # runs = infos_ap + infos_ft + infos_bt + infos_f
    runs = infos_ap + infos_ft + infos_f

    for wrapper_id in ["ScaleAction"]:
        for env_name in ["walker-walk-v0"]:
            for wrapper_dim in ["0"]:
                save_fname = f"figures/sequential__basic__{env_name}__wdim_{wrapper_dim}__wid_{wrapper_id}__transfer_summary"

                xlim = (min(runs[0]["xs"]), max(runs[0]["xs"]))
                ylim = (0.0, 1.05)
                bins = np.linspace(*xlim, 20 + 1, endpoint=True)

                tensor, filters, groups, tasks, seeds = plotting.tensor(runs, bins, bin=False)

                fig, axes = plotting.plots(len(filters), cols=3, xticks=2, grid=(2, 2), )

                for filtering, ax in zip(filters, axes):
                    title = filtering
                    ax.set_title(title)
                    # ax.set_xlabel("Action scaling")
                    # ax.set_ylabel(title)
                    ax.set_xlim(*xlim)
                    ax.set_ylim(*ylim)
                    ax.xaxis.set_major_formatter(plotting.smart_format)

                axes_id = 0
                for i, filtering in enumerate(filters):
                    ax = axes[axes_id]
                    axes_id += 1
                    color_id = 0
                    for j, grouping in reversed(list(enumerate(groups))):
                        for k, task in reversed(list(enumerate(tasks))):  # reversed for correct legend
                            # Aggregate over seeds.
                            mean = np.nanmean(tensor[i, j, k, :, :], axis=0)
                            std = np.nanstd(tensor[i, j, k, :, :], axis=0)
                            plotting.curve(
                                ax, bins, mean,
                                # ax, bins[1:], mean,
                                low=mean + std / 2,
                                high=mean - std / 2,
                                label=grouping,
                                order=j,
                                # color=plotting.COLORS_GRAY[color_id],
                                color=plotting.COLORS_GRAY[0],
                                linestyle=plotting.LINESTYLES[color_id]
                            )
                        color_id += 1
                if len(tasks) > 1:
                    from matplotlib.lines import Line2D
                    custom_legend_element = {
                        f"CTX {int(task.split('_')[-1]) + 1}": Line2D(
                            [0], [0],
                            color="k",
                            linestyle=plotting.LINESTYLES[k],
                        ) for k, task in enumerate(tasks)
                    }
                else:
                    custom_legend_element = None
                plotting.legend(
                    fig,
                    adjust=True,
                    custom_elements=custom_legend_element,
                    ncol=3,
                    plotpad=1.0,
                    handlelength=1.8,
                    title="algorithm",
                    legendpad=(0, -0.05)
                )
                plotting.save(fig, save_fname)


if __name__ == "__main__":
    make_plots_basic_selection()
    make_plots_transfer()
