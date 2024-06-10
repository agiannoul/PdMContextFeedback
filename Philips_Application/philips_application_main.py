import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import methods
import read_Data
import utils
import evaluation.evaluation as eval
from ContextApp.FalsePositives import PruneFPstream
from mango import scheduler, Tuner
import re

def plot_debug(start,end,events,names):
    colors=[np.random.rand(3,) for n in names]
    shown=[]
    for lista,name in zip(events,names):
        for ev in lista:
            if ev>start and ev<=end:
                if name in shown:
                    plt.axvline(ev,color=colors[names.index(name)])
                else:
                    plt.axvline(ev,label=name,color=colors[names.index(name)])
                    shown.append(name)





def run_full_pipeline(selftuning_window, smooth_median, profile_hours, function_reference, plot_them, method_params,
                      threshold_similarity, alpha, context_horizon, username="kr", savedist=False):

    list_train, list_test, names, event_lists,types,isfailure = read_Data.philips_semi_supervised(period_or_count=f"{profile_hours}")
    predictions_all = []
    dates_all = []
    failures = []
    dfall=None
    for dftrain, dftest in zip(list_train, list_test):
        scores = function_reference(dftrain, dftest,**method_params)
        scores = utils.self_tunning(scores, window_length=selftuning_window)
        scores = utils.moving_median(smooth_median, scores)
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
    allresults, results = eval.AUCPR_new(predictions_all, datesofscores=dates_all, PH="8 hours", lead="1 minutes",
                                         plot_them=plot_them, isfailure=isfailure)
    optimize = 0
    maxpos = 0
    for i in range(len(allresults)):
        if allresults[i][optimize] > allresults[maxpos][optimize]:
            maxpos = i
    threshold = allresults[maxpos][7]
    if plot_them:
        recall, Precision, f1, FPR = eval.pdm_eval_multi_PH(predictions_all,
                                                            [threshold for q in dates_all],
                                                            datesofscores=dates_all, isfailure=isfailure,
                                                            PH="8 hours", lead="1 minutes", plotThem=True)
        print("=== Before ===")
        print(f"FPR: {FPR}")
        print(f"f1: {f1[0]}")
        print(f"Precision: {Precision}")
        print(f"recall: {recall[0]}")
        print(f"best threshold: {threshold}")
    #return allresults[maxpos][optimize]
    ####################################################
    predictions = eval._flatten(predictions_all)




    prunner = PruneFPstream( dfall,  predictions, threshold, dfall.index, failures,contain_raw_data=False,
                 contextdata_list=event_lists,types=types,names=names,
                 threshold_similarity=threshold_similarity, alpha=alpha, context_horizon=f"{context_horizon} hours",
                 username=username, consider_FP="8 hours",savedistances=savedist)

    predicts = prunner.prune_scores()
    # back to original shape:
    counter = 0
    preditcions_after_prune = []
    for episode in dates_all:
        preditcions_after_prune.append([sc for sc in predicts[counter:counter + len(episode)]])
        counter = counter + len(episode)

    recall, Precision, f1,FPR = eval.pdm_eval_multi_PH(preditcions_after_prune, [0.5 for q in dates_all],
                                                   datesofscores=dates_all, isfailure=isfailure,
                                                   PH="8 hours", lead="1 minutes", plotThem=plot_them)

    if plot_them:
        print("=== AFTER ===")
        print(f"FPR: {FPR}")
        print(f"f1: {f1[0]}")
        print(f"Precision: {Precision}")
        print(f"recall: {recall[0]}")
    return f1[optimize]


@scheduler.parallel(n_jobs=3)
def optimization_objective(**params: dict):
    # print(params)
    username="Philips"
    if params["function_reference"] == "ocsvm":
        function_reference = methods.ocsvm_semi
        username+="_ocsvm"
    elif params["function_reference"] == "if":
        function_reference = methods.isolation_fores_semi
        username += "_if"
    elif params["function_reference"] == "kr":
        function_reference = methods.distance_based
        username += "_kr"
    elif params["function_reference"] == "lof":
        function_reference = methods.lof_semi
        username += "_lof"

    method_params = {re.sub('method_', '', k): v for k, v in params.items() if 'method' in k}

    print(params)
    #best_param:
    opt=run_full_pipeline(selftuning_window=params["selftuning_window"],
                          smooth_median=params["smooth_median"],
                          profile_hours=params["profile_hours"],
                          function_reference=function_reference, plot_them=False,
                          method_params=method_params,
                          username=username,
                          threshold_similarity=params["threshold_similarity"],
                          alpha=params["alpha"],
                          context_horizon=f"{params['context_horizon']} hours",
                          savedist=False)

    return opt


def pre_compute_contexts(param_space):
    username = "Philips"
    if param_space["function_reference"][0] == "ocsvm":
        function_reference = methods.ocsvm_semi
        username += "_ocsvm"
    elif param_space["function_reference"][0] == "if":
        function_reference = methods.isolation_fores_semi
        username += "_if"
    elif param_space["function_reference"][0] == "kr":
        function_reference = methods.distance_based
        username += "_kr"
    elif param_space["function_reference"][0] == "lof":
        function_reference = methods.lof_semi
        username += "_lof"

    method_params = {re.sub('method_', '', k): v[0] for k, v in param_space.items() if 'method' in k}

    for context_h in param_space["context_horizon"]:

        opt = run_full_pipeline(selftuning_window=param_space["selftuning_window"][0],
                                smooth_median=param_space["smooth_median"][0],
                                profile_hours=param_space["profile_hours"][0],
                                function_reference=function_reference, plot_them=False,
                                method_params=method_params,
                                username=username,
                                threshold_similarity=0.9,
                                alpha=param_space["alpha"][0],
                                context_horizon=f"{context_h} hours",
                                savedist=True)
        print(f"save {context_h}")
def run_Philips_Feedback_Application_mango(name):
    conf_dict = {
        'initial_random': 3,
        'num_iteration': 9 * 2,
    }

    if name == "ocsvm":
        param_space = {
            "profile_hours": [30],
            "smooth_median": [10],
            "selftuning_window": [40],
            "function_reference": ["ocsvm"],
            "threshold_similarity": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "alpha": [0, 0.25, 0.5, 0.75, 1],
            "context_horizon": [8, 16, 24],
        }
    elif name == "if":
        param_space = {
            "profile_hours": [30],
            "smooth_median": [5],
            "selftuning_window": [10],
            "function_reference": ["if"],
            "threshold_similarity": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            "alpha": [0, 0.25, 0.5, 0.75, 1],
            "context_horizon": [8, 16, 24],
        }
    elif name == "kr":
        param_space = {
            "profile_hours": [30],
            "smooth_median": [10],
            "selftuning_window": [60],
            "function_reference": ["kr"],
            "threshold_similarity": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            # "threshold_similarity": [0.3,0.35, 0.4,0.45, 0.5,0.55, 0.6,0.65, 0.7,0.75, 0.8,0.85, 0.9,0.95],
            "alpha": [0.25, 0.5, 0.75, 1],
            "context_horizon": [8, 16, 24],
        }
    elif name == "lof":
        param_space = {
            "profile_hours": [50],
            "smooth_median": [10],
            "selftuning_window": [40],
            "function_reference": ["lof"],
            "method_n_neighbors": [2],

            "threshold_similarity": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "alpha": [0, 0.25, 0.5, 0.75, 1],
            "context_horizon": [8, 16, 24],
        }
    pre_compute_contexts(param_space)



    tuner = Tuner(param_space, optimization_objective, conf_dict=conf_dict)
    results = tuner.maximize()
    print(f'Optimal value of parameters: {results["best_params"]} and objective: {results["best_objective"]}')



#
# if __name__ == '__main__':
#     opt = run_full_pipeline_without_cateforical(selftuning_window=60,
#                                                 smooth_median=10,
#                                                 profile_hours=30,
#                                                 function_reference=methods.distance_based,
#                                                 plot_them=True,
#                                                 method_params={},#{'n_neighbors': 2},
#                                                 username="Philips_kr",
#                                                 threshold_similarity=0.9,
#                                                 alpha=0.25,
#                                                 context_horizon=f"{8} hours",
#                                                 savedist=True)
#     #run_mango()