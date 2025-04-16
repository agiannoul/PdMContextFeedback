import pickle
import re
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

import read_Data
import methods
import evaluation.evaluation as eval
import utils
from mango import scheduler, Tuner

from ContextApp.FalsePositives import PruneFP


@scheduler.parallel(n_jobs=6)
def optimization_objective(**params: dict):
    print(params)
    username="Azure"
    if params["function_reference"] == "ocsvm":
        function_reference = methods.ocsvm_semi
        username += "_ocsvm"
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

    beta = params["beta"]
    if beta != 1 and beta != 1.0:
        username += f"_b_{beta}"
    # best_param:
    if params["add_raw_to_context"]:
        username += "_rawTrue"
    else:
        username += "_rawFalse"
    opt = run_full_pipeline(selftuning_window=params["selftuning_window"],
                                                smooth_median=params["smooth_median"],
                                                profile_hours=params["profile_hours"],
                                                function_reference=function_reference, plot_them=False,
                                                method_params=method_params,
                                                #username=username,
                                                username=None,
                                                threshold_similarity=params["threshold_similarity"],
                                                alpha=params["alpha"],
                                                context_horizon=f"{params['context_horizon']} hours",
                                                add_raw_to_context=params["add_raw_to_context"],
                                                beta=beta)

    return opt

def constraint_function(params_configuration_list):
    result = []

    for params in params_configuration_list:
        if params["selftuning_window"]+params["smooth_median"]+params["profile_hours"]>201:
            result.append(False)
        elif "method_n_neighbors" in params.keys():
            if params["method_n_neighbors"]>=params["profile_hours"]:
                result.append(False)
            else:
                result.append(True)
        else:
            result.append(True)

    return result

list_of_df = None
old_context_list_dfs = None
old_isfailure = None
old_all_sources = None


def initialize_Azure_data():
    global old_isfailure
    global old_context_list_dfs
    global old_all_sources
    global list_of_df

    list_of_df, old_context_list_dfs, old_isfailure, old_all_sources = read_Data.AzureData()


def run_detection(list_train,list_test,context_list_dfs,all_sources,function_reference,selftuning_window,smooth_median,method_params,beta,isfailure,profile_hours,plot_them):

    name = f"./Databases/AzureRuns/beta{beta}_pf{profile_hours}_sw{selftuning_window}_sm{smooth_median}_{function_reference.__name__}.pickle"
    #print(name)
    my_file = Path(name)
    if my_file.is_file():
        with open(name, 'rb') as f:
            tostore=pickle.load(f)
    else:
        print("Run and Save results for future use.")
        predictions_all = []
        dates_all = []
        failures = []
        counter = 0
        sources_data = {}
        sources_context = {}
        sources_failures = {}
        sources_predictions = {}
        for s in list(set(all_sources)):
            sources_data[s] = None
            sources_context[s] = None
            sources_failures[s] = []
            sources_predictions[s] = []
        for dftrain, dftest, dfcont, s in zip(list_train, list_test, context_list_dfs, all_sources):
            scores = function_reference(dftrain, dftest, **method_params)
            scores = utils.self_tunning(scores, window_length=selftuning_window)
            scores = utils.moving_median(smooth_median, scores)
            predictions_all.append(scores)
            sources_predictions[s].append(scores)
            # dates_all.append([dtt for dtt in dftest.index])
            dates_all.append([qi + counter for qi in range(len(dftest.index))])
            counter += len(dftest.index)

            # for prune:
            if sources_data[s] is None:
                sources_data[s] = dftest.copy()
                sources_context[s] = dfcont.loc[dftest.index[0]:].copy()

            else:
                sources_data[s] = pd.concat([sources_data[s], dftest])
                sources_context[s] = pd.concat([sources_context[s], dfcont.loc[dftest.index[0]:].copy()])

            failures.append(dftest.index[-1])
            sources_failures[s].append(dftest.index[-1])
        allresults, results = eval.AUCPR_new(predictions_all, datesofscores=dates_all, PH="96", lead="1",
                                             plot_them=False, isfailure=isfailure, beta=beta)
        optimize = 0
        maxpos = 0
        for i in range(len(allresults)):
            if allresults[i][optimize] > allresults[maxpos][optimize]:
                maxpos = i
        threshold = allresults[maxpos][7]

        tostore = {"predictions_all": predictions_all,
                   "dates_all": dates_all,
                   "sources_context": sources_context,
                   "sources_data": sources_data,
                   "sources_failures": sources_failures,
                   "sources_predictions": sources_predictions,
                   "threshold":threshold
                   }

        with open(name, 'wb') as f:
            pickle.dump(tostore, f)
    return tostore["predictions_all"],tostore["dates_all"],tostore["sources_data"],tostore["sources_context"],tostore["sources_failures"],tostore["sources_predictions"],tostore["threshold"]


def run_full_pipeline(selftuning_window,smooth_median,profile_hours,
                                          function_reference,plot_them,method_params,
                                          username,
                                          threshold_similarity,alpha,context_horizon,add_raw_to_context=True,beta=1):

    # list_of_df, context_list_dfs,isfailure,all_sources = read_Data.AzureData()
    global old_isfailure
    global old_context_list_dfs
    global old_all_sources
    global list_of_df
    list_train, list_test,context_list_dfs,isfailure,all_sources = read_Data.Azure_generate_train_test(list_of_df,old_isfailure,old_context_list_dfs,old_all_sources, period_or_count=f"{profile_hours} hours")
    predictions_all, dates_all, sources_data, sources_context, sources_failures, sources_predictions,threshold=run_detection(list_train,list_test,context_list_dfs,all_sources,function_reference,
                                                                                                                             selftuning_window,smooth_median,method_params,beta,isfailure,profile_hours,plot_them)
    if plot_them:
        recall, Precision, f1, FPR = eval.pdm_eval_multi_PH(predictions_all,
                                                            [threshold for q in dates_all],
                                                            datesofscores=dates_all, isfailure=isfailure,
                                                            PH="96", lead="1", plotThem=True)
        print("=== Before ===")
        print(f"FPR: {FPR}")
        print(f"f1: {f1[0]}")
        print(f"Precision: {Precision}")
        print(f"recall: {recall[0]}")
        print(f"best threshold: {threshold}")
    optimize = 0

    all_pruned_predicts=[]
    for s in sources_data.keys():
        #print(f"source: {s}")
        predictions = eval._flatten(sources_predictions[s])

        if username is None:
            source_username= None
        else:
            source_username=f"{username}_{s}"
        prunner = PruneFP(sources_data[s], sources_context[s], predictions, threshold, sources_data[s].index, sources_failures[s],
                          threshold_similarity=threshold_similarity, alpha=alpha, context_horizon=context_horizon,
                          username=source_username, consider_FP="96 hours",add_raw_to_context=add_raw_to_context)

        predicts = prunner.prune_scores()
        all_pruned_predicts.extend(predicts)
        #false_positives = prunner.getFalsepositives()
    # back to original shape:
    counter = 0
    preditcions_after_prune = []
    for episode in dates_all:
        preditcions_after_prune.append([sc for sc in all_pruned_predicts[counter:counter + len(episode)]])
        counter = counter + len(episode)

    recall, Precision, f1, FPR = eval.pdm_eval_multi_PH(preditcions_after_prune, [0.5 for q in dates_all],
                                                        datesofscores=dates_all, isfailure=isfailure,
                                                        PH="96", lead="1", plotThem=plot_them)

    if plot_them:
        print("=== AFTER ===")
        print(f"FPR: {FPR}")
        print(f"f1: {f1[0]}")
        print(f"Precision: {Precision}")
        print(f"recall: {recall[0]}")

        #plt.scatter(false_positives, [1 for i in false_positives])
        # for fail in failures:
        #     plt.axvline(fail)
        # plt.show()
    return f1[optimize]

def run_Azure_Feedback_Application_mango(params,beta=1,saved=False):
    conf_dict = {
        'initial_random': 3,
        'num_iteration': 6 * 3,
    }
    param_space ={
            "add_raw_to_context": [True,False],
            "threshold_similarity": [0.3,0.4,0.5,0.6,0.7,0.8,0.9],
            "alpha": [0,0.25,0.5,0.75,1],
            "context_horizon": [48, 96,192],
            "beta": [beta],
        }
    for pkey in params.keys():
        if pkey not in param_space.keys():
            param_space[pkey]=[params[pkey]]

    # if name == "ocsvm":
    #
    #     param_space = {
    #         "profile_hours": [50],
    #         "smooth_median": [10],
    #         "selftuning_window": [100],
    #         "function_reference": ["ocsvm"],
    #         "add_raw_to_context": [True, False],
    #         "threshold_similarity": [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    #         "alpha": [0,0.25,0.5,0.75,1],
    #         "context_horizon": [48, 96,192],
    #         "beta": [beta],
    #     }
    # elif name == "if":
    #     param_space = {
    #         "profile_hours": [100],
    #         "smooth_median": [10],
    #         "selftuning_window": [50],
    #         "function_reference": ["if"],
    #         "add_raw_to_context": [True,False],
    #         "threshold_similarity": [0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    #         "alpha": [0,0.25,0.5,0.75,1],
    #         "context_horizon": [48, 96,192],
    #         "beta": [beta],
    #     }
    # elif name == "kr":
    #     param_space = {
    #         "profile_hours": [150],
    #         "smooth_median": [10],
    #         "selftuning_window": [0],
    #         "function_reference": ["kr"],
    #         "add_raw_to_context": [True, False],
    #         "threshold_similarity": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         "alpha": [0, 0.25, 0.5, 0.75, 1],
    #         "context_horizon": [48, 96,192],
    #         "beta": [beta],
    #     }
    # elif name == "lof":
    #     param_space = {
    #         "profile_hours": [150],
    #         "smooth_median": [10],
    #         "selftuning_window": [0],
    #         "method_n_neighbors": [40],
    #         "function_reference": ["lof"],
    #         "add_raw_to_context": [True, False],
    #         "threshold_similarity": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         "alpha": [0, 0.25, 0.5, 0.75, 1],
    #         "context_horizon": [48, 96,192],
    #         "beta": [beta],
    #     }
    print(param_space)
    if saved:
        pre_compute_contexts(param_space)
    tuner = Tuner(param_space, optimization_objective, conf_dict=conf_dict)
    results = tuner.maximize()
    print(f'Optimal value of parameters: {results["best_params"]} and objective: {results["best_objective"]}')



def pre_compute_contexts(param_space):
    username = "Azure"
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

    beta=1
    if "beta" in param_space.keys():
        beta=param_space["beta"][0]
    if beta!=1 and beta != 1.0:
        username += f"_b_{beta}"
    tempusername=username
    for addraw in param_space["add_raw_to_context"]:
        if addraw:
            username = tempusername + "_rawTrue"
        else:
            username = tempusername + "_rawFalse"
        for context_h in param_space["context_horizon"]:
            run_full_pipeline(selftuning_window=param_space["selftuning_window"][0],
                                                  smooth_median=param_space["smooth_median"][0],
                                                  profile_hours=param_space["profile_hours"][0],
                                                  function_reference=function_reference, plot_them=False,
                                                  method_params=method_params,
                                                  username=username,
                                                  threshold_similarity=0.5,
                                                  alpha=0,
                                                  context_horizon=f"{context_h} hours",
                                                    add_raw_to_context=param_space["add_raw_to_context"][0],
                                                  beta=beta)














