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


def make_plots():
    for env_name in ["walker-walk-v0", "HalfCheetah-v4"]:
        for wrapper_dim in ["0", "None"]:

            base_dir = "../jaxrl3/multitask/"
            filter_dict = {
                "project_name": "DynamicsSwitch-multitask-0",
                "exp_type": "basic",
                "algorithm": "sac",
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
                filtering="wrapper_id"
            )
            for returns_type, runs in zip(["train", "eval"], [return_train, return_eval]):
                save_fname = f"figures/multitask__basic__{env_name}__wdim_{wrapper_dim}__wid_all__wval_all__{returns_type}"

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
                plotting.legend(fig, adjust=True)

                # Saves the figure in both PNG and PDF formats and attempts to crop margins off
                # the PDF.
                plotting.save(fig, save_fname)


def make_plots_mt_selection():
    for env_name in ["walker-walk-v0"]:
        for wrapper_dim in ["0", "None"]:

            base_dir = "../jaxrl3/multitask/"
            save_fname = f"figures/multitask__basic__selection__{env_name}__wdim_{wrapper_dim}"

            filter_selection = ["InvertAction"]
            grouping_selection = ["0.2"]

            fig, axes = plotting.plots(2 * len(filter_selection), cols=2, xticks=2, grid=(2, 2))

            for exp_type in ["reference_task_0", "reference", "basic"]:
                base_dir = "../jaxrl3/multitask/"
                filter_dict = {
                    "project_name": "DynamicsSwitch-multitask-0",
                    "exp_type": exp_type,
                    "algorithm": "sac",
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
                    filtering="wrapper_id"
                )

                xlim = (0, max(return_eval[0]["xs"]))
                bins = np.linspace(*xlim, 30 + 1, endpoint=True)

                tensor, filters, groups, tasks, seeds = plotting.tensor(return_eval, bins)

                axes_id = 0 if exp_type == "basic" else 1

                for filtering in filter_selection:
                    title = f"MT {filtering}" if exp_type == "basic" else f"ST {filtering}"
                    axes[axes_id].set_title(title)
                    axes[axes_id].set_xlim(*xlim)
                    axes[axes_id].xaxis.set_major_formatter(plotting.smart_format)

                color_id = 0
                for i, filtering in enumerate(filters):
                    if filtering in filter_selection:
                        ax = axes[axes_id]
                        for j, grouping in enumerate(groups):
                            if grouping in grouping_selection:
                                for k, task in enumerate(tasks):  # reversed for correct legend
                                    if exp_type == "reference" and k == 0:
                                        continue
                                    if exp_type == "reference_task_0" and k == 1:
                                        continue
                                    # Aggregate over seeds.
                                    mean = np.nanmean(tensor[i, j, k, :, :], axis=0)
                                    std = np.nanstd(tensor[i, j, k, :, :], axis=0)
                                    plotting.curve(
                                        ax, bins[1:], mean,
                                        low=mean + std / 2,
                                        high=mean - std / 2,
                                        label=f"Task {int(task.split('_')[-1]) + 1}",
                                        order=j,
                                        color=plotting.COLORS_GRAY[color_id],
                                        linestyle=plotting.LINESTYLES[k]
                                    )
                                color_id += 1
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
            plotting.legend(fig, adjust=True)

            # Saves the figure in both PNG and PDF formats and attempts to crop margins off
            # the PDF.
            plotting.save(fig, save_fname)


def compute_mt_parallel_transfer():
    infos = []
    for env_name in ["walker-walk-v0", "HalfCheetah-v4"]:
        for wrapper_dim in ["0", "None"]:

            tensor_exp = {}
            for exp_type in ["reference_task_0", "reference", "basic"]:
                base_dir = "../jaxrl3/multitask/"
                filter_dict = {
                    "project_name": "DynamicsSwitch-multitask-0",
                    "exp_type": exp_type,
                    "algorithm": "sac",
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
                    filtering="wrapper_id"
                )
                if exp_type in ["reference_task_0", "reference", "basic"]:
                    ats = [0] + [int(len(return_eval[0]["xs"]) / 2 - 1)] + [-1]
                else:
                    raise NotImplementedError()
                tensor, filters, groups, tasks, seeds = plotting.tensor_at(return_eval, ats)
                tensor_exp[exp_type] = tensor

            assert len(tasks) == 2
            for i, filtering in enumerate(filters):
                for j, grouping in enumerate(groups):
                    max_over_mt_tasks = np.max(np.nanmean(tensor_exp["basic"][i, j, :, :, :], axis=1))
                    max_over_ref0_tasks = np.max(np.nanmean(tensor_exp["reference_task_0"][i, j, :, :, :], axis=1))
                    max_over_ref1_tasks = np.max(np.nanmean(tensor_exp["reference"][i, j, :, :, :], axis=1))
                    max_over_tasks = np.max([max_over_mt_tasks, max_over_ref0_tasks, max_over_ref1_tasks])

                    diff_task_0 = (tensor_exp["basic"][i, j, 0, :, :] - tensor_exp["reference_task_0"][i, j, 0, :, :]) / max_over_tasks
                    diff_task_1 = (tensor_exp["basic"][i, j, 1, :, :] - tensor_exp["reference"][i, j, 1, :, :]) / max_over_tasks

                    mean_diff_over_tasks = np.nanmean([diff_task_0, diff_task_1], axis=0)

                    # Aggregate over seeds.
                    mean = np.nanmean(mean_diff_over_tasks, axis=0)
                    std = np.nanstd(mean_diff_over_tasks, axis=0)
                    info = {}
                    info["env_name"] = env_name
                    info["wrapper_dim"] = wrapper_dim
                    info["filter"] = filtering
                    info["group"] = grouping
                    info["mean"] = np.around(mean, 3)
                    info["std"] = np.around(std / 2, 3)
                    infos.append(info)
    return infos


def compute_mt_avg_performance():
    infos = []
    for env_name in ["walker-walk-v0", "HalfCheetah-v4"]:
        for wrapper_dim in ["0", "None"]:

            tensor_exp = {}
            for exp_type in ["basic"]:
                base_dir = "../jaxrl3/multitask/"
                filter_dict = {
                    "project_name": "DynamicsSwitch-multitask-0",
                    "exp_type": exp_type,
                    "algorithm": "sac",
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
                    filtering="wrapper_id"
                )
                if exp_type in ["basic"]:
                    ats = [0] + [int(len(return_eval[0]["xs"]) / 2 - 1)] + [-1]
                else:
                    raise NotImplementedError()
                tensor, filters, groups, tasks, seeds = plotting.tensor_at(return_eval, ats)
                tensor_exp[exp_type] = tensor

            assert len(tasks) == 2
            for i, filtering in enumerate(filters):
                for j, grouping in enumerate(groups):
                    max_over_tasks = np.max(np.nanmean(tensor_exp["basic"][i, j, :, :, :], axis=1))

                    mean_over_tasks = np.nanmean(tensor_exp["basic"][i, j, :, :, :] / max_over_tasks, axis=0)

                    # Aggregate over seeds.
                    mean = np.nanmean(mean_over_tasks, axis=0)
                    std = np.nanstd(mean_over_tasks, axis=0)
                    info = {}
                    info["env_name"] = env_name
                    info["wrapper_dim"] = wrapper_dim
                    info["filter"] = filtering
                    info["group"] = grouping
                    info["mean"] = np.around(mean, 3)
                    info["std"] = np.around(std / 2, 3)
                    infos.append(info)
    return infos


def make_tables():
    infos_ap = compute_mt_avg_performance()
    infos_pt = compute_mt_parallel_transfer()

    # keep last value for avg performance
    for i, info in enumerate(infos_ap):
        infos_ap[i]["mean"] = infos_ap[i]["mean"][-1]
        infos_ap[i]["std"] = infos_ap[i]["std"][-1]
    # keep last value for parallel transfer
    for i, info in enumerate(infos_ap):
        infos_pt[i]["mean"] = infos_pt[i]["mean"][-1]
        infos_pt[i]["std"] = infos_pt[i]["std"][-1]

    for algorithm in ["sac"]:
        for env_name in ["walker-walk-v0", "HalfCheetah-v4"]:
            dfs = []
            for metric_name, metric_infos in zip(["Avg. performance", "Parallel transfer"], [infos_ap, infos_pt]):
                df = pd.DataFrame(metric_infos)
                # df = df.loc[df["algorithm"] == algorithm]
                df = df.loc[df["env_name"] == env_name]
                df = df.round(2)
                df = df.loc[~((df["filter"].isin(["InvertAction", "SwapAction"])) & (df["group"].isin(["0.1", "0.2", "0.3", "0.4", "0.6", "0.7", "0.8", "0.9"])))]
                df = df.loc[df["group"].isin(["0.2", "0.5", "0.8"])]
                df = df.reset_index(drop=True)
                df.loc[df["filter"].isin(["InvertAction", "SwapAction"]), "group"] = "-"
                df.loc[df["wrapper_dim"] == "None", "wrapper_dim"] = "all"
                df[metric_name] = df["mean"].astype(str) + " (" + df["std"].astype(str) + ")"
                df = df.drop(columns=["mean", "std", "env_name"])
                df = df.set_index(["filter", "wrapper_dim", "group"])
                df.index.names = ["Wrapper", "Dim.", "Value"]
                df = df.sort_index(level="Wrapper")
                dfs.append(df)

            df = pd.concat(dfs, axis=1)

            base_dir = "tables/"
            os.makedirs(base_dir, exist_ok=True)
            fname = base_dir + f"multitask__basic__{env_name}__{algorithm}.tex"

            df.to_latex(fname, multirow=True)

            # custom modifications
            with open(fname, 'r') as file:
                # read a list of lines into data
                data = file.readlines()

            data[0] = '\\begin{tabular}{lllll}\n'
            data[2] = '\\multicolumn{1}{c}{\\bf Wrapper}  &\\multicolumn{1}{c}{\\bf Dim.}  &\\multicolumn{1}{c}{\\bf Value}  &\\multicolumn{1}{c}{\\bf Avg. performance}  &\\multicolumn{1}{c}{\\bf Parallel transfer}\n'
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
    make_plots()
    make_plots_mt_selection()
    make_tables()
    # infos_pt = compute_mt_parallel_transfer()
    # infos_ap = compute_mt_avg_performance()
