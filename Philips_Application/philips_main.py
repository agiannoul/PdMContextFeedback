import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import read_Data
import methods
import evaluation.evaluation as eval
import utils
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




def run_full_pipeline(selftuning_window, smooth_median, profile_hours, function_reference, plot_them, method_params):

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
        print(f"before : {allresults[maxpos][optimize]}")
    return allresults[maxpos][optimize]


def constraint_function(params_configuration_list):
    result = []

    for params in params_configuration_list:
        if params["selftuning_window"]+params["smooth_median"]+params["profile_hours"]>200:
            result.append(False)
        elif "method_n_neighbors" in params.keys():
            if params["method_n_neighbors"]>=params["profile_hours"]:
                result.append(False)
            else:
                result.append(True)
        else:
            result.append(True)

    return result


def run_Philips_main_mango(name):
    conf_dict = {
        'initial_random': 3,
        'num_iteration': 6 * 3,
        'constraint': constraint_function
    }

    #        "method_n_neighbors": [2,10,15,20,30,40],
    if name ==  "ocsvm":
        param_space = {
            "profile_hours": [30, 50, 100, 200],
            "smooth_median": [1, 3, 5, 10],
            "selftuning_window": [1, 10, 20, 24, 30, 40, 60],
            "function_reference": ["ocsvm"],
        }
    elif name == "if":
        param_space = {
            "profile_hours": [30, 50, 100, 200],
            "smooth_median": [1, 3, 5, 10],
            "selftuning_window": [1, 10, 20, 24, 30, 40, 60],
            "function_reference": ["if"],
        }
    elif name == "kr":
            param_space = {
                "profile_hours": [30,50, 100, 200],
                "smooth_median": [1, 3, 5, 10],
                "selftuning_window": [1, 10, 20, 24, 30, 40, 60],
                "function_reference": ["kr"],
            }
    elif name == "lof":
        param_space = {
            "profile_hours": [30, 50, 100, 200],
            "smooth_median": [1, 3, 5, 10],
            "selftuning_window": [1, 10, 20, 24, 30, 40, 60],
            "function_reference": ["lof"],
            "method_n_neighbors": [2, 10, 15, 20, 30, 40, 60],
        }
    tuner = Tuner(param_space, optimization_objective, conf_dict=conf_dict)
    results = tuner.maximize()
    print(f'Optimal value of parameters: {results["best_params"]} and objective: {results["best_objective"]}')


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



    opt_met=run_full_pipeline(selftuning_window=params["selftuning_window"], smooth_median=params["smooth_median"],
                              profile_hours=params["profile_hours"],
                              function_reference=function_reference, plot_them=False, method_params=method_params)


    #print(opt_met)
    return opt_met

