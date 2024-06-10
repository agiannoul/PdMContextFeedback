import re

import read_Data
import methods
import evaluation.evaluation as eval
import utils
from mango import scheduler, Tuner



@scheduler.parallel(n_jobs=6)
def optimization_objective(**params: dict):
    #print(params)
    if params["function_reference"]=="ocsvm":
        function_reference = methods.ocsvm_semi
    elif params["function_reference"]=="if":
        function_reference = methods.isolation_fores_semi
    elif params["function_reference"]=="kr":
        function_reference = methods.distance_based
    elif params["function_reference"]=="lof":
        function_reference = methods.lof_semi

    method_params = {re.sub('method_', '', k): v for k, v in params.items() if 'method' in k}

    beta=params["beta"]

    opt_met=run_full_pipeline_without_cateforical(selftuning_window=params["selftuning_window"],
                                        smooth_median=params["smooth_median"],
                                          profile_hours=params["profile_hours"],
                                          function_reference=function_reference,
                                          plot_them=False,method_params=method_params,
                                         beta=beta)
    #print(opt_met)
    return opt_met

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





list_of_df, old_context_list_dfs,old_isfailure,old_all_sources = read_Data.AzureData()



def run_full_pipeline_without_cateforical(selftuning_window,smooth_median,
                                          profile_hours,function_reference,
                                          plot_them,method_params,
                                          beta=1):
    global old_isfailure
    global old_context_list_dfs
    global old_all_sources
    global list_of_df
    list_train, list_test, context_list_dfs, isfailure, all_sources = read_Data.Azure_generate_train_test(list_of_df,
                                                                                                          old_isfailure,
                                                                                                          old_context_list_dfs,
                                                                                                          old_all_sources,
                                                                                                          period_or_count=f"{profile_hours} hours")

    predictions_all = []
    dates_all = []
    counter=0
    for dftrain, dftest in zip(list_train, list_test):
        scores = function_reference(dftrain, dftest,**method_params)
        scores = utils.self_tunning(scores, window_length=selftuning_window)
        scores = utils.moving_median(smooth_median, scores)
        predictions_all.append(scores)
        #dates_all.append([dtt for dtt in dftest.index])
        dates_all.append([qi+counter for qi in range(len(scores))])
        counter+=len(scores)
    allresults, results = eval.AUCPR_new(predictions_all, datesofscores=dates_all, PH="96", lead="1",
                                         plot_them=False, isfailure=isfailure,beta=beta)




    optimize = 0
    maxpos = 0
    for i in range(len(allresults)):
        if allresults[i][optimize] > allresults[maxpos][optimize]:
            maxpos = i
    if plot_them:
        recall, Precision, f1, FPR = eval.pdm_eval_multi_PH(predictions_all, [allresults[maxpos][7] for q in dates_all],
                                                            datesofscores=dates_all, isfailure=isfailure,
                                                            PH="96", lead="1", plotThem=True)
        print("=== Before ===")
        print(f"FPR: {FPR}")
        print(f"f1: {f1[0]}")
        print(f"Precision: {Precision}")
        print(f"recall: {recall[0]}")
        print(f"best threshold: {allresults[maxpos][7]}")
    return allresults[maxpos][optimize]



def run_Azure_main_mango(name,beta=1):
    conf_dict = {
        'initial_random': 3,
        'num_iteration': 6 * 3,
        'constraint': constraint_function
    }
    param_space = {
        "profile_hours": [10, 50, 100, 150, 200],
        "smooth_median": [0, 3, 5, 10],
        "selftuning_window": [0, 10, 25, 50, 100],
        "beta" : [beta],
    }
    if name ==  "ocsvm":
        param_space["function_reference"]=["ocsvm"]
    elif name == "if":
        param_space["function_reference"]=["if"]
    elif name == "kr":
        param_space["function_reference"]=["kr"]
    elif name == "lof":
        param_space["function_reference"] = ["lof"]
        param_space["method_n_neighbors"]= [2, 10, 15, 20, 30, 40]

    tuner = Tuner(param_space, optimization_objective, conf_dict=conf_dict)
    results = tuner.maximize()
    print(f'Optimal value of parameters: {results["best_params"]} and objective: {results["best_objective"]}')
    return results["best_params"]

















