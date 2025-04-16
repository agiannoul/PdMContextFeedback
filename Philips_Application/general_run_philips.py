import os
import re

import pandas as pd
from matplotlib import pyplot as plt

import evaluation.evaluation as eval
import methods
import read_Data
from ContextApp.FalsePositives import PruneFPstream
import utils

def run_full_pipeline(selftuning_window, smooth_median, profile_hours, function_reference, plot_them, method_params,
                      threshold_similarity, alpha, context_horizon, username="kr", savedist=False,beta_initial=0):
    PHregion="8 hours"
    FPregion="8 hours"
    list_train, list_test, names, event_lists,types,isfailure = read_Data.philips_semi_supervised(period_or_count=f"{profile_hours}")

    predictions_all = []
    dates_all = []
    failures = []
    dfall=None
    for dftrain, dftest in zip(list_train, list_test):
        scores = function_reference(dftrain, dftest,**method_params)
        scores = utils.self_tunning(scores, window_length=selftuning_window)
        scores = utils.moving_median(smooth_median, scores)
        # mms=max(scores)
        # mmin=min(scores)
        # scores=[(s-mmin)/(mms-mmin) for s in scores]
        # plt.plot(dftest.index,scores)
        # plot_debug(dftrain.index[0], dftest.index[-1], event_lists, names)
        # plt.legend()
        # plt.show()

        predictions_all.append(scores)
        dates_all.append([dtt for dtt in dftest.index])
        if dfall is None:
            dfall=dftest.copy()
        else:
            dfall=pd.concat([dfall,dftest])
        failures.append(dftest.index[-1])
    allresults, _ = eval.AUCPR_new(predictions_all, datesofscores=dates_all, PH=PHregion, lead="1 minutes",
                                         plot_them=DEBUG, isfailure=isfailure,beta=beta_initial)
    optimize = 0
    maxpos = 0
    for i in range(len(allresults)):
        if allresults[i][optimize] > allresults[maxpos][optimize]:
            maxpos = i
    threshold = allresults[maxpos][7]

    recall, Precision, f1, FPR = eval.pdm_eval_multi_PH(predictions_all,
                                                            [threshold for q in dates_all],
                                                            datesofscores=dates_all, isfailure=isfailure,
                                                            PH=PHregion, lead="1 minutes", plotThem=False)


    if plot_them:
        print("=== Before ===")
        print(f"FPR: {FPR}")
        print(f"f1: {f1[0]}")
        print(f"Precision: {Precision}")
        print(f"recall: {recall[0]}")
        print(f"best threshold: {threshold}")
    pre_best_recall=recall[0]
    pre_best_Precision=Precision
    pre_f1=f1[0]

    #return allresults[maxpos][optimize]
    ####################################################
    predictions = eval._flatten(predictions_all)




    prunner = PruneFPstream( dfall,  predictions, threshold, dfall.index, failures,contain_raw_data=False,
                 contextdata_list=event_lists,types=types,names=names,
                 threshold_similarity=threshold_similarity, alpha=alpha, context_horizon=f"{context_horizon} hours",
                 username=username, consider_FP=FPregion,savedistances=savedist,checkincrease=True)

    predicts = prunner.prune_scores()
    # back to original shape:
    counter = 0
    preditcions_after_prune = []
    for episode in dates_all:
        preditcions_after_prune.append([sc for sc in predicts[counter:counter + len(episode)]])
        counter = counter + len(episode)

    recall, Precision, f1,FPR = eval.pdm_eval_multi_PH(preditcions_after_prune, [0.5 for q in dates_all],
                                                   datesofscores=dates_all, isfailure=isfailure,
                                                   PH=PHregion, lead="1 minutes", plotThem=DEBUG)
    after_f1=f1[0]
    after_recall=recall[0]
    after_Precision=Precision
    if plot_them:
        print("=== AFTER ===")
        print(f"FPR: {FPR}")
        print(f"f1: {f1[0]}")
        print(f"Precision: {Precision}")
        print(f"recall: {recall[0]}")
    return f1[optimize],pre_f1,pre_best_recall,after_f1,after_recall,pre_best_Precision,after_Precision


def run_philips(param_space,save_dist=False,plot_them=True,restart=False):
    ## use method_ for method parameters

    method_params = {re.sub('method_', '', k): v for k, v in param_space.items() if 'method' in k}

    if param_space["function_reference"] == "ocsvm":
        function_reference = methods.ocsvm_semi
    elif param_space["function_reference"] == "if":
        function_reference = methods.isolation_fores_semi
    elif param_space["function_reference"] == "kr":
        function_reference = methods.distance_based
    elif param_space["function_reference"] == "lof":
        function_reference = methods.lof_semi
    elif param_space["function_reference"] == "deepAnt":
        function_reference = methods.lof_semi

    username = "philips"
    if param_space["function_reference"] == "ocsvm":
        function_reference = methods.ocsvm_semi
        username += "_ocsvm"
    elif param_space["function_reference"] == "if":
        function_reference = methods.isolation_fores_semi
        username += "_if"
    elif param_space["function_reference"] == "kr":
        function_reference = methods.distance_based
        username += "_kr"
    elif param_space["function_reference"] == "lof":
        function_reference = methods.lof_semi
        username += "_lof"
    elif param_space["function_reference"] == "deepAnt":
        function_reference = methods.deepAnt
        username += "_deepAnt"

    if restart:
        pre_s=f"ch_{param_space['context_horizon']}_hours_hours"
        file_path=f"Databases/stream/{pre_s}_{username}.db"
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File deleted {file_path}.")
        else:
            print(f"File not found {file_path}")
        file_path = f"Databases/stream/distances/{pre_s}_alpha_{param_space['alpha']}_{username}.pickle"
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File deleted {file_path}.")
        else:
            print(f"File not found {file_path}")

    return run_full_pipeline(
        selftuning_window=param_space["selftuning_window"],
        smooth_median=param_space["smooth_median"],
        profile_hours=param_space["profile_hours"],
        function_reference=function_reference,
        plot_them=plot_them,
        method_params=method_params,
        username=username,
        threshold_similarity=param_space['threshold_similarity'],
        alpha=param_space["alpha"],
        context_horizon=f"{param_space['context_horizon']} hours",
        savedist=save_dist,
        beta_initial=param_space["beta_initial"])

def run_experiment_philips(method_name=None,similarities=None,profile_hours=350,smooth_median=20,selftuning_window=100,alpha=0.25,beta_initial=4,context_horizon=8):
    # for techname in ["ocsvm", "if", "kr", "lof","deepAnt"]:
    restart=True
    if method_name is None:
        to_run=["deepAnt"]
    else:
        to_run=[method_name]
    if similarities is None:
        x = [0.2, 0.3, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9]
    else:
        x = [similarities]
    for techname in to_run:
        # x = [ 0.5, 0.6, 0.7, 0.8, 0.9]

        for similarity in x:
            param_space = {
                "profile_hours": profile_hours,
                "smooth_median": smooth_median,
                "selftuning_window": selftuning_window,
                "function_reference": techname,
                "threshold_similarity": similarity,
                "alpha": alpha,
                "context_horizon": context_horizon,
                "beta_initial": beta_initial,

            }
            if techname == "deepAnt":
                param_space["method_window_size"] = 10
            elif techname == "lof":
                param_space["method_n_neighbors"] = 2
            # elif techname == "if":
            #     param_space["method_random_state"] = 42
            print(f"==== {techname} ==== ")
            opt,pre_f1,pre_best_recall,after_f1,after_recall,pre_best_Precision,after_Precision = run_philips(param_space,save_dist=True,plot_them=False,restart=restart)
            restart = False
            print(after_recall)
            print(f"==== {techname} s={similarity}==== ")
            print(f"Pre F1: {f_beta_score(1, pre_best_recall, pre_best_Precision)}")
            print(f"After F1: {f_beta_score(1, after_recall, after_Precision)}")
            print(f"Pre F2: {f_beta_score(2, pre_best_recall, pre_best_Precision)}")
            print(f"After F2: {f_beta_score(2, after_recall, after_Precision)}")
            print(f"Pre F3: {f_beta_score(3, pre_best_recall, pre_best_Precision)}")
            print(f"After F3: {f_beta_score(3, after_recall, after_Precision)}")
            print(f"Pre_Recall: {pre_best_recall}")
            print(f"After recall: {after_recall}")
            print(f"Pre Precision: {pre_best_Precision}")
            print(f"After Precision: {after_Precision}")
            print(f"============================= ")

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


# if __name__ == "__main__":
#     run_experiment_philips()

DEBUG=False