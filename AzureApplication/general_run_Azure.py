import os
import pickle
import re
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from ContextApp.FalsePositives import PruneFP

import evaluation.evaluation as eval
import methods
import read_Data
from ContextApp.FalsePositives import PruneFPstream
import utils


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

list_of_df, old_context_list_dfs,old_isfailure,old_all_sources = read_Data.AzureData()


def run_full_pipeline(selftuning_window, smooth_median, profile_hours, function_reference, plot_them, method_params,
                      threshold_similarity, alpha, context_horizon, username="kr", savedist=False,beta_initial=0,add_raw_to_context=True):
    optimize = 0
    global old_isfailure
    global old_context_list_dfs
    global old_all_sources
    global list_of_df
    list_train, list_test, context_list_dfs, isfailure, all_sources = read_Data.Azure_generate_train_test(list_of_df,
                                                                                                          old_isfailure,
                                                                                                          old_context_list_dfs,
                                                                                                          old_all_sources,
                                                                                                          period_or_count=f"{profile_hours} hours")
    predictions_all, dates_all, sources_data, sources_context, sources_failures, sources_predictions, threshold = run_detection(
        list_train, list_test, context_list_dfs, all_sources, function_reference,
        selftuning_window, smooth_median, method_params, beta_initial, isfailure, profile_hours, plot_them)

    recall, Precision, f1, FPR = eval.pdm_eval_multi_PH(predictions_all,
                                                            [threshold for q in dates_all],
                                                            datesofscores=dates_all, isfailure=isfailure,
                                                            PH="96", lead="1", plotThem=False)


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

    all_pruned_predicts = []
    for s in sources_data.keys():
        # print(f"source: {s}")
        predictions = eval._flatten(sources_predictions[s])

        if username is None:
            source_username = None
        else:
            source_username = f"{username}_{s}"
        prunner = PruneFP(sources_data[s], sources_context[s], predictions, threshold, sources_data[s].index,
                          sources_failures[s],
                          threshold_similarity=threshold_similarity, alpha=alpha, context_horizon=context_horizon,
                          username=source_username, consider_FP="96 hours", add_raw_to_context=add_raw_to_context,
                          checkincrease=True)

        predicts = prunner.prune_scores()
        all_pruned_predicts.extend(predicts)
    # back to original shape:
    counter = 0
    preditcions_after_prune = []
    for episode in dates_all:
        preditcions_after_prune.append([sc for sc in all_pruned_predicts[counter:counter + len(episode)]])
        counter = counter + len(episode)

    recall, Precision, f1,FPR = eval.pdm_eval_multi_PH(preditcions_after_prune, [0.5 for q in dates_all],
                                                   datesofscores=dates_all, isfailure=isfailure,
                                                   PH="96", lead="1",  plotThem=DEBUG)
    after_f1=f1[0]
    after_recall=recall[0]
    if plot_them:
        print("=== AFTER ===")
        print(f"FPR: {FPR}")
        print(f"f1: {f1[0]}")
        print(f"Precision: {Precision}")
        print(f"recall: {recall[0]}")
    return f1[optimize],pre_f1,pre_best_recall,after_f1,after_recall,pre_best_Precision,Precision


def run_Azure(param_space,save_dist=False,plot_them=True,restart=False):
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

    username = "Azure"
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
    tempusername = username
    addraw = param_space["add_raw_to_context"]
    if addraw:
        username = tempusername + "_rawTrue"
    else:
        username = tempusername + "_rawFalse"
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
        beta_initial=param_space["beta_initial"],
        add_raw_to_context=False)

def run_experiment_Azure():
    # for techname in ["ocsvm", "if", "kr", "lof","deepAnt"]:
    restart=False
    for techname in ["ocsvm", "if", "kr", "lof","deepAnt"]:
        # x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        x = [0.99,0.995,0.999]

        # x = [0.8, 0.9, 0.95,0.975,0.99]
        for similarity in x:

            if techname=="kr":
                params = {'add_raw_to_context': False, 'alpha': 0.5, 'beta': 1, 'context_horizon': 48,
                      'function_reference': 'kr', 'profile_hours': 150, 'selftuning_window': 0, 'smooth_median': 10,
                      'threshold_similarity': 0.8}
            elif techname=="if":
                params = {'add_raw_to_context': False, 'alpha': 0.5, 'beta': 1, 'context_horizon': 48,
                      'function_reference': 'if', 'profile_hours': 100, 'selftuning_window': 50, 'smooth_median': 10,
                      'threshold_similarity': 0.6}
            elif techname == "ocsvm":
                params = {'add_raw_to_context': False, 'alpha': 0, 'beta': 1, 'context_horizon': 48,
                      'function_reference': 'ocsvm', 'profile_hours': 50, 'selftuning_window': 100, 'smooth_median': 10,
                      'threshold_similarity': 0.4}
            elif techname == "ocsvm":
                params = {'add_raw_to_context': False, 'alpha': 0, 'beta': 1, 'context_horizon': 48,
                      'function_reference': 'lof', 'method_n_neighbors': 40, 'profile_hours': 150,
                      'selftuning_window': 0, 'smooth_median': 10, 'threshold_similarity': 0.8}
            elif techname == "deepAnt":
                params = {'add_raw_to_context': False, 'alpha': 0.25, 'beta': 1, 'context_horizon': 48,
                      'function_reference': 'deepAnt', 'method_n_neighbors': 40, 'profile_hours': 150,
                      'selftuning_window': 0, 'smooth_median': 10, 'threshold_similarity': 0.8,"method_window_size":10}


            params[ "threshold_similarity"]= similarity
            params[ "beta_initial"]= 1
            param_space = params
            print(f"==== {techname} ==== ")
            opt,pre_f1,pre_best_recall,after_f1,after_recall , pre_best_Precision, after_Precision = run_Azure(param_space,save_dist=True,plot_them=False,restart=restart)
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

# if __name__ == "__main__":
#     run_experiment_philips()

DEBUG=False
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
#
# ==== deepAnt ====
# === Before ===
# FPR: 0.10323574062025591
# f1: 0.3344813369384609
# Precision: 0.20118363054617053
# recall: 0.9912536443148688
# best threshold: 1.2144993543624878
# === AFTER ===
# FPR: 0.021450155425431938
# f1: 0.4479101817958821
# Precision: 0.3271655328798186
# recall: 0.7099125364431487
# 0.7099125364431487
# 0.4479101817958821
# =============================
# ==== deepAnt ====
# === Before ===
# FPR: 0.10323574062025591
# f1: 0.3344813369384609
# Precision: 0.20118363054617053
# recall: 0.9912536443148688
# best threshold: 1.2144993543624878
# === AFTER ===
# FPR: 0.021574495770982433
# f1: 0.4498920752389763
# Precision: 0.3283521627582482
# recall: 0.7142857142857143
# 0.7142857142857143
# 0.4498920752389763
# =============================
# ==== deepAnt ====
# === Before ===
# FPR: 0.10323574062025591
# f1: 0.3344813369384609
# Precision: 0.20118363054617053
# recall: 0.9912536443148688
# best threshold: 1.2144993543624878
# === AFTER ===
# FPR: 0.022147039687703318
# f1: 0.46321633938713286
# Precision: 0.339485145099392
# recall: 0.7288629737609329
# 0.7288629737609329
# 0.46321633938713286
# =============================
# ==== deepAnt ====
# === Before ===
# FPR: 0.10323574062025591
# f1: 0.3344813369384609
# Precision: 0.20118363054617053
# recall: 0.9912536443148688
# best threshold: 1.2144993543624878
# === AFTER ===
# FPR: 0.022745608327911514
# f1: 0.4671949915601692
# Precision: 0.3400176196669044
# recall: 0.7463556851311953
# 0.7463556851311953
# 0.4671949915601692
# =============================
# ==== deepAnt ====
# === Before ===
# FPR: 0.10323574062025591
# f1: 0.3344813369384609
# Precision: 0.20118363054617053
# recall: 0.9912536443148688
# best threshold: 1.2144993543624878
# === AFTER ===
# FPR: 0.02279332032097159
# f1: 0.46675106163888597
# Precision: 0.3395475492249686
# recall: 0.7463556851311953
# 0.7463556851311953
# 0.46675106163888597
# =============================
# ==== deepAnt ====
# === Before ===
# FPR: 0.10323574062025591
# f1: 0.3344813369384609
# Precision: 0.20118363054617053
# recall: 0.9912536443148688
# best threshold: 1.2144993543624878
# === AFTER ===
# FPR: 0.022807778500686764
# f1: 0.4669273833911915
# Precision: 0.3394330220677526
# recall: 0.7478134110787172
# 0.7478134110787172
# 0.4669273833911915
# =============================
# ==== deepAnt ====
# === Before ===
# FPR: 0.10323574062025591
# f1: 0.3344813369384609
# Precision: 0.20118363054617053
# recall: 0.9912536443148688
# best threshold: 1.2144993543624878
# === AFTER ===
# FPR: 0.0228106701366298
# f1: 0.46690048866833467
# Precision: 0.3394045974123854
# recall: 0.7478134110787172
# 0.7478134110787172
# 0.46690048866833467
# =============================
# ==== deepAnt ====
# === Before ===
# FPR: 0.10323574062025591
# f1: 0.3344813369384609
# Precision: 0.20118363054617053
# recall: 0.9912536443148688
# best threshold: 1.2144993543624878
# === AFTER ===
# FPR: 0.0228106701366298
# f1: 0.46690048866833467
# Precision: 0.3394045974123854
# recall: 0.7478134110787172
# 0.7478134110787172
# 0.46690048866833467
# =============================



################# OPTIMIZE F1:
#
# ==== deepAnt ====
# === Before ===
# FPR: 0.05045181811609918
# f1: 0.3645693560815418
# Precision: 0.22938474449008436
# recall: 0.8877551020408163
# best threshold: 1.4394757151603699
# === AFTER ===
# FPR: 0.009115882310417118
# f1: 0.21655430970078887
# Precision: 0.24072736030828518
# recall: 0.1967930029154519
# 0.1967930029154519
# 0.21655430970078887
# =============================
# ==== deepAnt ====
# === Before ===
# FPR: 0.05045181811609918
# f1: 0.3645693560815418
# Precision: 0.22938474449008436
# recall: 0.8877551020408163
# best threshold: 1.4394757151603699
# === AFTER ===
# FPR: 0.013807561627991035
# f1: 0.447084824460913
# Precision: 0.3391003460207612
# recall: 0.6559766763848397
# 0.6559766763848397
# 0.447084824460913
# =============================
# ==== deepAnt ====
# === Before ===
# FPR: 0.05045181811609918
# f1: 0.3645693560815418
# Precision: 0.22938474449008436
# recall: 0.8877551020408163
# best threshold: 1.4394757151603699
# === AFTER ===
# FPR: 0.01382201980770621
# f1: 0.4514662506886105
# Precision: 0.34376716090060405
# recall: 0.6574344023323615
# 0.6574344023323615
# 0.4514662506886105
# =============================
# ==== deepAnt ====
# === Before ===
# FPR: 0.05045181811609918
# f1: 0.3645693560815418
# Precision: 0.22938474449008436
# recall: 0.8877551020408163
# best threshold: 1.4394757151603699
# === AFTER ===
# FPR: 0.014059133955035061
# f1: 0.461661235820887
# Precision: 0.3516036540641462
# recall: 0.6720116618075802
# 0.6720116618075802
# 0.461661235820887
# =============================
# ==== deepAnt ====
# === Before ===
# FPR: 0.05045181811609918
# f1: 0.3645693560815418
# Precision: 0.22938474449008436
# recall: 0.8877551020408163
# best threshold: 1.4394757151603699
# === AFTER ===
# FPR: 0.014167570302898865
# f1: 0.4617650095690659
# Precision: 0.3509306484732066
# recall: 0.6749271137026239
# 0.6749271137026239
# 0.4617650095690659
# =============================
# ==== deepAnt ====
# === Before ===
# FPR: 0.05045181811609918
# f1: 0.3645693560815418
# Precision: 0.22938474449008436
# recall: 0.8877551020408163
# best threshold: 1.4394757151603699
# === AFTER ===
# FPR: 0.014630232053784428
# f1: 0.46074882483683505
# Precision: 0.3482125603864734
# recall: 0.6807580174927114
# 0.6807580174927114
# 0.46074882483683505
# =============================
# ==== deepAnt ====
# === Before ===
# FPR: 0.05045181811609918
# f1: 0.3645693560815418
# Precision: 0.22938474449008436
# recall: 0.8877551020408163
# best threshold: 1.4394757151603699
# === AFTER ===
# FPR: 0.015859177329574207
# f1: 0.45200648163135976
# Precision: 0.33690001209043646
# recall: 0.6865889212827988
# 0.6865889212827988
# 0.45200648163135976
# =============================
# ==== deepAnt ====
# === Before ===
# FPR: 0.05045181811609918
# f1: 0.3645693560815418
# Precision: 0.22938474449008436
# recall: 0.8877551020408163
# best threshold: 1.4394757151603699
# === AFTER ===
# FPR: 0.01956191715463023
# f1: 0.4281982202508037
# Precision: 0.30877694901399816
# recall: 0.6982507288629738
# 0.6982507288629738
# 0.4281982202508037
# =============================