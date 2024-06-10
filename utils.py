import statistics
import numpy as np

def self_tunning(scores: list[float],window_length) -> list[float]:
    if window_length<=1:
        return scores
    scores_for_calculating_metrics = scores[:window_length]
    if len(scores_for_calculating_metrics) > 1:
        mean, std = statistics.mean(scores_for_calculating_metrics), np.std(scores_for_calculating_metrics)
    else:
        mean = statistics.mean(scores_for_calculating_metrics)
        std = 0

    if std == 0.0:
        return [sc - mean for sc in scores]

    return list(map(lambda score: (score - mean) / std, scores))


def moving_median(window,data):
    if window<=1:
        return data
    median_series=[]
    for i in range(window,len(data)):
        median_series.append(statistics.median(data[i-window:i]))
    toreturn = [median_series[0] for i in range(window)]
    toreturn.extend(median_series)
    return toreturn
def moving_mean(window,data):
    median_series=[]
    for i in range(window,len(data)):
        median_series.append(statistics.mean(data[i-window:i]))
    toreturn = [median_series[0] for i in range(window)]
    toreturn.extend(median_series)
    return toreturn