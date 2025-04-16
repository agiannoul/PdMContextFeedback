import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import methods
import read_Data
import utils
import evaluation.evaluation as eval
from ContextApp.FalsePositives import PruneFPstream
import re

def run_full_pipeline_without_cateforical(selftuning_window,smooth_median,profile_hours,function_reference,plot_them,method_params,
                                          threshold_similarity,alpha,context_horizon,username="kr",printhem=False):

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
    if printhem:
        print(" =========== BEFORE =======")
        print(f"f1 : {allresults[maxpos][0]}")
        print(f"recall: {allresults[maxpos][3]}")
        print(f"Precision: {allresults[maxpos][6]}")
        print(f"FPR: {allresults[maxpos][8]}")

    #return allresults[maxpos][optimize]
    ####################################################
    predictions = eval._flatten(predictions_all)




    prunner = PruneFPstream( dfall,  predictions, threshold, dfall.index, failures,contain_raw_data=False,
                 contextdata_list=event_lists,types=types,names=names,
                 threshold_similarity=threshold_similarity, alpha=alpha, context_horizon=f"{context_horizon} hours",
                 username=f"{username}", consider_FP="8 hours",savedistances=True)

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

    if printhem:
        print(" =========== AFTER =======")
        print(f"f1 : {f1[optimize]}")
        print(f"recall: {recall[optimize]}")
        print(f"Precision: {Precision}")
        print(f"FPR: {FPR}")
    return f1[optimize]







def optimization_objective(params: dict,printhem=False):
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

    #best_param:
    opt=run_full_pipeline_without_cateforical(selftuning_window=params["selftuning_window"],
                                                    smooth_median=params["smooth_median"],
                                                    profile_hours=params["profile_hours"],
                                                    function_reference=function_reference, plot_them=False,
                                                    method_params=method_params,
                                                    username=username,
                                                    threshold_similarity=params["threshold_similarity"],
                                                    alpha=params["alpha"],
                                                    context_horizon=f"{params['context_horizon']} hours",
                                                                             printhem=printhem)

    return opt


def similarity_PB():
    keep_opts = []
    for similarity in [0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        param_space = {
            "profile_hours": 30,
            "smooth_median": 10,
            "selftuning_window": 60,
            "function_reference": "kr",
            "threshold_similarity": similarity,
            "alpha": 0.75,
            "context_horizon": 8,
        }

        opt = optimization_objective(param_space)
        print(opt)
        keep_opts.append(opt)
    print(keep_opts)
    plt.plot(keep_opts)
    plt.plot([0.26 for i in keep_opts])
    plt.show()

def similarity_IF():
    keep_opts = []
    for similarity in [0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        param_space = {
            "profile_hours": 30,
            "smooth_median": 5,
            "selftuning_window": 10,
            "function_reference": "if",
            "threshold_similarity": similarity,
            "alpha": 0.25,
            "context_horizon": 24,
        }

        opt = optimization_objective(param_space)
        print(opt)
        keep_opts.append(opt)
    print(keep_opts)
    plt.plot(keep_opts)
    plt.plot([0.25 for i in keep_opts])
    plt.show()



def similarity_ocsvm():
    keep_opts = []
    for similarity in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        param_space = {
            "profile_hours": 30,
            "smooth_median": 10,
            "selftuning_window": 40,
            "function_reference": "ocsvm",
            "threshold_similarity": similarity,
            "alpha": 0,
            "context_horizon": 24,
        }

        opt = optimization_objective(param_space)
        print(opt)
        keep_opts.append(opt)
    print(keep_opts)
    plt.plot(keep_opts)
    plt.plot([0.17 for i in keep_opts])
    plt.show()

def similarity_lof():
    keep_opts = []
    for similarity in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        param_space = {
            "profile_hours": 50,
            "smooth_median": 10,
            "selftuning_window": 40,
            "function_reference": "lof",
            "threshold_similarity": similarity,
            'method_n_neighbors': 2,
            "alpha": 0,
            "context_horizon": 8,

        }

        opt = optimization_objective(param_space)
        print(opt)
        keep_opts.append(opt)
    print(keep_opts)
    plt.plot(keep_opts)
    plt.plot([0.17 for i in keep_opts])
    plt.show()

def plot_inner(x,y,line,name="(a) ocsvm"):
    linew = 5
    marksize = 20
    plt.plot(x, y, ".-", color="black", linewidth=linew, markersize=marksize)
    plt.axhline(line, color="grey", linewidth=linew)
    plt.xlabel(f"similarity threshold \n {name}")
    plt.ylabel("F1-score")
    plt.ylim([0,0.6])
    plt.grid(True)


def plot_nice():
    x = [0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # y = [0.3988803358992302,0.3787375415282392, 0.3306264501160093, 0.30760928224500805, 0.2961038961038961, 0.17762542848239327,
    #  0.17762542848239327, 0.17762542848239327]
    y=[0.22043010752688175, 0.18637532133676094, 0.17750826901874311, 0.17723880597014924, 0.17723880597014924, 0.17723880597014924, 0.17723880597014924, 0.17723880597014924]

    line = 0.17
    plt.subplot(321)
    plot_inner(x, y, line, name="(a) OCSVM")


    x=[0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #y=[0.5134228187919463,0.5464909155682223, 0.5260631001371742, 0.49918646273999356, 0.3952589538778666, 0.373134328358209, 0.36860879904875143, 0.36539368222536533]
    y =[0.3055555555555556, 0.33064516129032256, 0.47088186356073214, 0.5201407012368092, 0.4667549129416556, 0.41723479683665127, 0.423991155334439, 0.4007314524555904]


    line=0.26
    plt.subplot(323)
    plot_inner(x,y,line,name="(c) pb")

    x = [0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #y = [0.3431952662721893,0.04738562091503268, 0.2795941375422773, 0.2713347921225383, 0.26373626373626374, 0.259958071278826, 0.259958071278826, 0.259958071278826]
    y = [0.08668730650154799, 0.08668730650154799, 0.08408408408408408, 0.327718223583461, 0.327718223583461, 0.327718223583461, 0.4073319755600815, 0.34917891097666376]


    line = 0.25
    plt.subplot(322)
    plot_inner(x, y, line, name="(b) IF")

    x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #y = [0.5396109388215394, 0.590134529147982, 0.5043534343760078, 0.47537993920972643, 0.4364731653888281, 0.3801574051991415, 0.2974990668159761, 0.2960074280408542]
    y = [0.5220910623946037, 0.5168614357262103, 0.48374999999999996, 0.4287280701754386, 0.42755604155276106, 0.4117956819378621, 0.40750390828556543, 0.40750390828556543]

    line = 0.27
    plt.subplot(324)
    plot_inner(x, y, line, name="(d) LOF")

    x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # y = [0.5396109388215394, 0.590134529147982, 0.5043534343760078, 0.47537993920972643, 0.4364731653888281, 0.3801574051991415, 0.2974990668159761, 0.2960074280408542]
    y = [0.45,0.426,0.393,0.356,0.335,0.320,0.319,0.307]

    line = 0.237
    plt.subplot(325)
    plot_inner(x, y, line, name="(3) DeepAnt")

    plt.show()

def get_recall_precission_FPR_PB():
    param_space = {
        "profile_hours": 30,
        "smooth_median": 10,
        "selftuning_window": 60,
        "function_reference": "kr",
        "threshold_similarity": 0.5,
        "alpha": 0.75,
        "context_horizon": 8,
    }

    opt = optimization_objective(param_space,printhem=True)


def get_recall_precission_FPR_LOF():
    param_space = {
        "profile_hours": 50,
        "smooth_median": 10,
        "selftuning_window": 40,
        "function_reference": "lof",
        "threshold_similarity": 0.2,
        'method_n_neighbors': 2,
        "alpha": 0,
        "context_horizon": 8,

    }

    opt = optimization_objective(param_space,printhem=True)



def get_recall_precission_FPR_OCSVM():
    param_space = {
        "profile_hours": 30,
        "smooth_median": 10,
        "selftuning_window": 40,
        "function_reference": "ocsvm",
        "threshold_similarity": 0.2,
        "alpha": 0,
        "context_horizon": 24,
    }

    opt = optimization_objective(param_space,printhem=True)


def get_recall_precission_FPR_IF():
    param_space = {
        "profile_hours": 30,
        "smooth_median": 5,
        "selftuning_window": 10,
        "function_reference": "if",
        "threshold_similarity": 0.85,
        "alpha": 0.75,
        "context_horizon": 8,
    }

    opt = optimization_objective(param_space,printhem=True)


#if __name__ == '__main__':

