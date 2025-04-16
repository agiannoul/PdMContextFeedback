import re
import numpy as np

import matplotlib.pyplot as plt

def plot_resbars_F2(resbars, Study,ax,field="F2"):
    techniques = list(resbars.keys())
    num_techniques = len(techniques)
    bar_width = 0.3  # Width of each bar
    x = np.arange(num_techniques)  # Positions for techniques

    # Create subplots for F2

    # Prepare data for F2
    f2_before = [resbars[tech][field][1] for tech in techniques]
    f2_after = [resbars[tech][field][0] for tech in techniques]

    # Plot bars for F2
    ax.bar(x - bar_width / 2, f2_before, bar_width,label=f"Before", color="steelblue", edgecolor="black")
    ax.bar(x + bar_width / 2, f2_after, bar_width, label=f"After", color="lightsteelblue", edgecolor="black")

    # Add labels, title, and legend
    ax.set_xlabel("Techniques", fontsize=14)
    ax.set_ylabel("F2-score", fontsize=12)
    ax.set_title(f"{Study} case.", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    # Add gridlines
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(fontsize=14,prop={'weight': 'bold','size': 14},ncol=2)

    # Save the plot as a high-resolution image
    # plt.savefig("performance_plot.png", dpi=300)


def plot_resbars_F05(resbars, Study,ax,field="F0.5"):
    techniques = list(resbars.keys())
    num_techniques = len(techniques)
    bar_width = 0.3  # Width of each bar
    x = np.arange(num_techniques)  # Positions for techniques

    # Create subplots for F2

    # Prepare data for F2
    f2_before = [resbars[tech][field][1] for tech in techniques]
    f2_after = [resbars[tech][field][0] for tech in techniques]

    # Plot bars for F2
    ax.bar(x - bar_width / 2, f2_before, bar_width, label=f"Before", color="darkgreen", edgecolor="black")
    ax.bar(x + bar_width / 2, f2_after, bar_width, label=f"After", color="lightgreen", edgecolor="black")

    # Add labels, title, and legend
    ax.set_xlabel("Techniques", fontsize=14)
    ax.set_ylabel("F0.5-score", fontsize=12)
    ax.set_title(f"{Study} case.", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    # Add gridlines
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(fontsize=14,prop={'weight': 'bold','size': 14},ncol=2)

    # Save the plot as a high-resolution image
    # plt.savefig("performance_plot.png", dpi=300)
def f_beta_score(beta, recall, precision):
    """
    Calculate the F-beta score.

    Parameters:
        beta (float): Weight of recall in the combined score.
        recall (float): Recall value (between 0 and 1).
        precision (float): Precision value (between 0 and 1).

    Returns:
        float: F-beta score.
    """
    if recall == 0 and precision == 0:
        return 0.0

    beta_squared = beta ** 2
    return (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)

def plot_resbars(resbars,Study):
    techniques = list(resbars.keys())
    metrics = ["F2", "F3"]
    num_techniques = len(techniques)
    bar_width = 0.2  # Width of each bar
    x = np.arange(num_techniques)  # Positions for techniques

    # Create subplots for F2 and F3
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for F2 and F3
    f2_before = [resbars[tech]["F2"][1] for tech in techniques]
    f2_after = [resbars[tech]["F2"][0] for tech in techniques]
    f3_before = [resbars[tech]["F3"][1] for tech in techniques]
    f3_after = [resbars[tech]["F3"][0] for tech in techniques]

    # Plot bars for F2
    ax.bar(x - bar_width / 2, f2_before, bar_width, label="F2 Before", color="steelblue", edgecolor="black")
    ax.bar(x + bar_width / 2, f2_after, bar_width, label="F2 After", color="lightsteelblue", edgecolor="black")

    # Plot bars for F3
    ax.bar(x + bar_width * 1.5, f3_before, bar_width, label="F3 Before", color="darkgreen", edgecolor="black")
    ax.bar(x + bar_width * 2.5, f3_after, bar_width, label="F3 After", color="lightgreen", edgecolor="black")

    # Add labels, title, and legend
    ax.set_xlabel("Techniques", fontsize=14)
    ax.set_ylabel("Scores", fontsize=14)
    ax.set_title(f"Performance difference for F2-scores and F3-score on {Study} case.", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, fontsize=12)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(fontsize=12)

    # Add gridlines
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the plot as a high-resolution image
    plt.tight_layout()
    # plt.savefig("performance_plot.png", dpi=300)
    plt.show()

def read_metrics(name,folder="azureRRR"):
    with open(f"Databases/{folder}/{name}", "r") as file:
        data = file.read()

    metrics = {
        "Pre F1": [],
        "After F1": [],
        "Pre F2": [],
        "After F2": [],
        "Pre F3": [],
        "After F3": [],
        "Pre_Recall": [],
        "After recall": [],
        "Pre Precision": [],
        "After Precision": []
    }

    pattern = re.compile(
        r"(Pre F1|After F1|Pre F2|After F2|Pre F3|After F3|Pre_Recall|After recall|Pre Precision|After Precision): ([0-9.]+)")

    # Process each match and populate the lists
    for match in re.finditer(pattern, data):
        key, value = match.groups()
        metrics[key].append(float(value))
    return metrics

def F2_F3(folder="azureRRR",study="Azure",ax=None):
    resbars={}

    for name,techname in zip(["rif", "rocsvm", "rpb", "rlof", "rdeepAnt"],["IF", "OCSVM", "PB", "Lof", "DeepAnt"]):
        metrics=read_metrics(name,folder)

        # plt.plot(metrics["After recall"], metrics["After Precision"])
        # plt.scatter(metrics["Pre_Recall"], metrics["Pre Precision"])
        # plt.show()

        mxF2=max(metrics["After F2"])
        mxF3=max(metrics["After F3"])
        prF2 = max(metrics["Pre F2"])
        prF3 = max(metrics["Pre F3"])

        prF05 = max([f_beta_score(0.5,rc,pr) for rc,pr in zip(metrics["Pre_Recall"], metrics["Pre Precision"])])
        mxF05 = max([f_beta_score(0.5,rc,pr) for rc,pr in zip(metrics["After recall"], metrics["After Precision"])])
        prF1 = max([f_beta_score(1,rc,pr) for rc,pr in zip(metrics["Pre_Recall"], metrics["Pre Precision"])])
        mxF1 = max([f_beta_score(1,rc,pr) for rc,pr in zip(metrics["After recall"], metrics["After Precision"])])
        resbars[techname]={}
        resbars[techname]["F2"] = [mxF2, prF2]
        resbars[techname]["F3"] = [mxF3, prF3]
        resbars[techname]["F0.5"] = [mxF05, prF05]
        resbars[techname]["F1"] = [mxF1, prF1]

    # plot_resbars(resbars,Study=study)
    plot_resbars_F2(resbars,Study=study,ax=ax)
    # plot_resbars_F05(resbars,Study=study,ax=ax)

def F2_F3_all():
    resbars={}
    for folder in ["philipsRRR","azureRRR","metroRRR"]:
        for name,techname in zip(["rif", "rocsvm", "rpb", "rlof", "rdeepAnt"],["IF", "OCSVM", "PB", "Lof", "DeepAnt"]):
            metrics=read_metrics(name,folder)

            # plt.plot(metrics["After recall"], metrics["After Precision"])
            # plt.scatter(metrics["Pre_Recall"], metrics["Pre Precision"])
            # plt.show()

            mxF2=max(metrics["After F2"])
            mxF3=max(metrics["After F3"])
            prF2 = max(metrics["Pre F2"])
            prF3 = max(metrics["Pre F3"])

            prF05 = max([f_beta_score(0.5,rc,pr) for rc,pr in zip(metrics["Pre_Recall"], metrics["Pre Precision"])])
            mxF05 = max([f_beta_score(0.5,rc,pr) for rc,pr in zip(metrics["After recall"], metrics["After Precision"])])
            prF1 = max([f_beta_score(1,rc,pr) for rc,pr in zip(metrics["Pre_Recall"], metrics["Pre Precision"])])
            mxF1 = max([f_beta_score(1,rc,pr) for rc,pr in zip(metrics["After recall"], metrics["After Precision"])])
            resbars[techname+folder]={}
            resbars[techname+folder]["F2"] = [mxF2, prF2]
            resbars[techname+folder]["F3"] = [mxF3, prF3]
            resbars[techname+folder]["F0.5"] = [mxF05, prF05]
            resbars[techname+folder]["F1"] = [mxF1, prF1]
    get_statistical_signif_(resbars, "F1")
    print("===================")
    get_statistical_signif_(resbars, "F2")
    print("===================")
    get_statistical_signif_(resbars, "F0.5")



def get_statistical_signif_(resbars,metric):
    from scipy.stats import wilcoxon
    After = []
    Before = []
    average_diff = []
    average_improvment = []
    for key in resbars.keys():
        After.append(resbars[key][metric][0])
        Before.append(resbars[key][metric][1])
        average_diff.append(resbars[key][metric][0]-resbars[key][metric][1])
        average_improvment.append((resbars[key][metric][0]-resbars[key][metric][1])/resbars[key][metric][1])
    res = wilcoxon(After, Before, alternative='greater')
    if res[1] <= 0.05:
        print(f"After {metric} is statistically significant greater.")
    else:
        res = wilcoxon(After, Before, alternative='less')
        if res[1] <= 0.05:
            print(f"After {metric} is statistically significant less.")
        else:
            print(f"After {metric} is not statistically significant different.")
    print(f"Average difference: {np.mean(average_diff)}")
    print(f"Average improvement: {np.mean(average_improvment)}")
    print(f"Average positive improvement: {np.mean([av for av in average_improvment if av>=0])}")
    print(f"wins: {len([av for av in average_improvment if av>0])}")



F2_F3_all()

fig, ax = plt.subplots(figsize=(10, 6),nrows=1,ncols=3)


F2_F3(folder="philipsRRR",study="Philips",ax=ax[0])
F2_F3(folder="azureRRR",study="Azure",ax=ax[1])
F2_F3(folder="metroRRR",study="Metro",ax=ax[2])
plt.show()


