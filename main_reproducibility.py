import re

import methods
from Philips_Application import philips_application_main
from Philips_Application.general_run_philips import run_experiment_philips
from Philips_Application.philips_application_main import run_Philips_Feedback_Application_mango
from Philips_Application.philips_main import run_Philips_main_mango
from metroapplication.test_new_dataset import run_metro


def fine_tunne_methods_in_Azure_datasets():
    from AzureApplication.AzureMain import run_Azure_main_mango

    # names : "lof", "kr" for Profile Based, "if", "ocsvm"
    # categorical: whether categorical values will be considered or no.

    params=run_Azure_main_mango("kr", beta=1)
    return params

def fine_tunne_in_Azure_high_recall():
    from AzureApplication.AzureMain import run_Azure_main_mango

    params=run_Azure_main_mango("kr", beta=1)
    return params


def fine_tunne_Feedback_in_Azure_high_recall(params):
    from AzureApplication.AzureFeedback import run_Azure_Feedback_Application_mango

    run_Azure_Feedback_Application_mango(params, beta=3)

def fine_tunne_Feedback_in_Azure_datasets(params):
    """
    After calculating the best parameters using the fine_tunne_methods_in_Azure_datasets()
    and passed them to run_Metro_Feedback_Application_mango params.
    :return:
    """
    from AzureApplication.AzureFeedback import run_Azure_Feedback_Application_mango

    run_Azure_Feedback_Application_mango(params)

def fine_tunne_methods_in_Philips_datasets():
    # names : "lof", "kr", "if", "ocsvm"
    # categorical: whether categorical values will be considered or no.
    run_Philips_main_mango(name="kr")

def fine_tunne_Feedback_in_Philips_datasets():
    """
    After calculating the best parameters using the fine_tunne_methods_in_Philips_datasets()
    and passed them to run_Metro_Feedback_Application_mango params.
    :return:
    """
    #Optimal value of parameters: {'alpha': 0.75, 'context_horizon': 8, 'function_reference': 'kr', 'profile_hours': 30, 'selftuning_window': 60, 'smooth_median': 10, 'threshold_similarity': 0.5} and objective: 0.5201407012368092

    #Optimal value of parameters: {'alpha': 0, 'context_horizon': 24, 'function_reference': 'ocsvm', 'profile_hours': 30, 'selftuning_window': 40, 'smooth_median': 10, 'threshold_similarity': 0.2} and objective: 0.2204301075268817

    #Optimal value of parameters: {'alpha': 0.25, 'context_horizon': 24, 'function_reference': 'if', 'profile_hours': 30, 'selftuning_window': 10, 'smooth_median': 5, 'threshold_similarity': 0.85} and objective: 0.4073319755600815

    #Optimal value of parameters: {'alpha': 0, 'context_horizon': 8, 'function_reference': 'lof', 'method_n_neighbors': 2, 'profile_hours': 50, 'selftuning_window': 40, 'smooth_median': 10, 'threshold_similarity': 0.2} and objective: 0.5220910623946037

    run_Philips_Feedback_Application_mango(name="lof")

def run_single_Philips(param_space):
    if param_space["function_reference"] == "ocsvm":
        function_reference = methods.ocsvm_semi
    elif param_space["function_reference"] == "if":
        function_reference = methods.isolation_fores_semi
    elif param_space["function_reference"] == "kr":
        function_reference = methods.distance_based
    elif param_space["function_reference"] == "lof":
        function_reference = methods.lof_semi

    method_params = {re.sub('method_', '', k): v for k, v in param_space.items() if 'method' in k}


    philips_application_main.run_full_pipeline(
                            selftuning_window=param_space["selftuning_window"],
                            smooth_median=param_space["smooth_median"],
                            profile_hours=param_space["profile_hours"],
                            function_reference=function_reference,
                            plot_them=True,
                            method_params=method_params,
                            username=None,
                            threshold_similarity=param_space['threshold_similarity'],
                            alpha=param_space["alpha"],
                            context_horizon=f"{param_space['context_horizon']} hours",
                            savedist=False)




def Azure_run_single(params):
    from AzureApplication import AzureFeedback

    if params["function_reference"] == "ocsvm":
        function_reference = methods.ocsvm_semi
    elif params["function_reference"] == "if":
        function_reference = methods.isolation_fores_semi
    elif params["function_reference"] == "kr":
        function_reference = methods.distance_based
    elif params["function_reference"] == "lof":
        function_reference = methods.lof_semi
    method_params = {re.sub('method_', '', k): v for k, v in params.items() if 'method' in k}

    AzureFeedback.run_full_pipeline(  # Window to use for thresholding technqiue
        selftuning_window=params['selftuning_window'],
        # a window for smoothing scores as post-processor
        smooth_median=params['smooth_median'],
        # How many hours of data after failure to use
        # for training in semi supervised detectors
        profile_hours=params['profile_hours'],
        # reference to the method used to produce scores
        function_reference=function_reference,
        plot_them=True,
        # pass parameters to the method
        method_params=method_params,
        # this is used in case we want to save calculated contexts
        username=None,
        # The similarity threshold over context
        threshold_similarity=params['threshold_similarity'],
        # parameter for similarity measurement
        alpha=0,
        # Time window of context
        context_horizon=f"{params['context_horizon']} hours",
        # This is true in case we want to use the raw data
        # on which the detector running in the context as well,
        add_raw_to_context=params["add_raw_to_context"],
        # which beta to use to calculate threshold for detector before context
        # the final evaluation is done in F1 score.
        beta=params['beta'])



def run_philips_AD_with_Feedback():
    run_experiment_philips(method_name="deepAnt", profile_hours=350, smooth_median=20, selftuning_window=100, alpha=0.25,
                           beta_initial=4,context_horizon=8)
    # pass similarities!=None e.g. similarities=0.3 for similarity threshold to prune false alarms.
    run_experiment_philips(method_name="if", profile_hours=30, smooth_median=5, selftuning_window=10,
                           alpha=0.75,
                           beta_initial=1,context_horizon=8)
    run_experiment_philips(method_name="ocsvm", profile_hours=30, smooth_median=10, selftuning_window=40,
                           alpha=0,
                           beta_initial=1,context_horizon=24)
    run_experiment_philips(method_name="lof", profile_hours=50, smooth_median=10, selftuning_window=40,
                           alpha=0,
                           beta_initial=2,context_horizon=8)
    run_experiment_philips(method_name="kr", profile_hours=30, smooth_median=10, selftuning_window=60,
                           alpha=0.75,
                           beta_initial=1, context_horizon=8)

def run_Azure_with_AD():
    from AzureApplication.general_run_Azure import run_experiment_Azure
    from AzureApplication.AzureFeedback import initialize_Azure_data

    initialize_Azure_data()
    run_experiment_Azure()

def run_Metro_with_AD():
    from metroapplication.test_new_dataset import run_experiment
    # this run all AD algorithms on multiple similarity thresholds
    run_experiment()

if __name__ == '__main__':
    run_philips_AD_with_Feedback()
    # run_Metro_with_AD()
    #run_Azure_with_AD()
