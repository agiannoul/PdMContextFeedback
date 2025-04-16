from matplotlib import pyplot as plt

import methods


def plot_inner(x,y,line,name="(a) ocsvm"):
    linew = 5
    marksize = 20
    plt.plot(x, y, ".-", color="black", linewidth=linew, markersize=marksize)
    plt.axhline(line, color="grey", linewidth=linew)
    plt.xlabel(f"similarity threshold \n {name}")
    plt.ylabel("F1-score")
    plt.ylim([0.3,0.6])
    plt.grid(True)

def plot_nice():

    x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    y=[0.4748064336006148, 0.5022528904401087, 0.5310689140987033, 0.5345276564352619, 0.5350995289537671, 0.5356787020478225, 0.5347796643128212]
    line=0.45
    plt.subplot(321)
    plot_inner(x,y,line,name="(a) PB")

    x = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #y = [0.5808383233532934, 0.5823529411764706, 0.5823529411764706, 0.5823529411764706, 0.5823529411764706, 0.5823529411764706, 0.5823529411764706]
    y = [0.4484182169384069, 0.4684082411836734, 0.49149475236941853, 0.4940262630107543, 0.48195479509196304, 0.46224754792704614, 0.44278005726337205]
    line = 0.42
    plt.subplot(322)
    plot_inner(x, y, line, name="(b) IF")

    x = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #y = [0.37755102040816324, 0.620253164556962, 0.6270270270270271, 0.6270270270270271, 0.6270270270270271,
    #     0.6270270270270271, 0.6270270270270271]
    y = [0.4469137306709569, 0.45762687431761084, 0.4572442371072609, 0.4572442371072609, 0.4572442371072609, 0.4572442371072609, 0.4572442371072609]
    line =0.41
    plt.subplot(323)
    plot_inner(x, y, line, name="(c) OCSVM")


    x = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    y = [0.4916709210135394, 0.5012484597781354, 0.5038043679705585, 0.5038043679705585, 0.5038043679705585, 0.5038043679705585, 0.5038043679705585]

    line = 0.45
    plt.subplot(324)
    plot_inner(x, y, line, name="(d) lof")

    x = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    y = [0.44,0.451,0.46, 0.46, 0.460,0.452,0.428]

    line = 0.36
    plt.subplot(325)
    plot_inner(x, y, line, name="(e) DeepAnt")
    plt.show()



def sensetivity():
    from AzureApplication import AzureFeedback

    # kr : [0.4748064336006148, 0.5022528904401087, 0.5310689140987033, 0.5345276564352619, 0.5350995289537671, 0.5356787020478225, 0.5347796643128212]
    # if : [0.4484182169384069, 0.4684082411836734, 0.49149475236941853, 0.4940262630107543, 0.48195479509196304, 0.46224754792704614, 0.44278005726337205]
    # ocsvm : [0.4469137306709569, 0.45762687431761084, 0.4572442371072609, 0.4572442371072609, 0.4572442371072609, 0.4572442371072609, 0.4572442371072609]
    # lof : [0.4916709210135394, 0.5012484597781354, 0.5038043679705585, 0.5038043679705585, 0.5038043679705585, 0.5038043679705585, 0.5038043679705585]
    params= {'add_raw_to_context': False, 'alpha': 0.5, 'beta': 1, 'context_horizon': 48, 'function_reference': 'kr', 'profile_hours': 150, 'selftuning_window': 0, 'smooth_median': 10, 'threshold_similarity': 0.8}
    params=  {'add_raw_to_context': False, 'alpha': 0.5, 'beta': 1, 'context_horizon': 48, 'function_reference': 'if', 'profile_hours': 100, 'selftuning_window': 50, 'smooth_median': 10, 'threshold_similarity': 0.6}
    params= {'add_raw_to_context': False, 'alpha': 0, 'beta': 1, 'context_horizon': 48, 'function_reference': 'ocsvm', 'profile_hours': 50, 'selftuning_window': 100, 'smooth_median': 10, 'threshold_similarity': 0.4}
    params= {'add_raw_to_context': False, 'alpha': 0, 'beta': 1, 'context_horizon': 48, 'function_reference': 'lof', 'method_n_neighbors': 40, 'profile_hours': 150, 'selftuning_window': 0, 'smooth_median': 10, 'threshold_similarity': 0.8}

    if params["function_reference"] == "ocsvm":
        function_reference = methods.ocsvm_semi
    elif params["function_reference"] == "if":
        function_reference = methods.isolation_fores_semi
    elif params["function_reference"] == "kr":
        function_reference = methods.distance_based
    elif params["function_reference"] == "lof":
        function_reference = methods.lof_semi

    fones=[]
    for threshold_sim in [0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        obj=AzureFeedback.run_full_pipeline(selftuning_window=params['selftuning_window'],
                                                         smooth_median=params['smooth_median'],
                                                  profile_hours=params['profile_hours'],
                                                  function_reference=function_reference,
                                                  plot_them=False,method_params={"n_neighbors":params["method_n_neighbors"]},
                                                  username=None,
                                                  threshold_similarity=threshold_sim,
                                                  alpha=params['alpha'],
                                                  context_horizon=f"{params['context_horizon']} hours",
                                                  #add_raw_to_context=params["add_raw_to_context"],
                                                  add_raw_to_context=params['add_raw_to_context'],
                                                  beta=params['beta'])
        fones.append(obj)
        print(obj)
    print(fones)