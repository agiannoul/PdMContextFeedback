import collections

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd

import subprocess
import os
import uuid

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from evaluation.vus.metrics import get_metrics

def calculate_ts_recall(anomalyranges, predictionsforrecall):
    generated_uuid = uuid.uuid4()
    anomaly_ranges_file_name = f'anomaly_ranges{generated_uuid}.real'
    anomaly_ranges_df = pd.DataFrame(anomalyranges, columns=['Numbers'])
    anomaly_ranges_df.to_csv(anomaly_ranges_file_name, index=False, header=False)

    prediction_file_name = f'anomaly_ranges{generated_uuid}.pred'
    predictions_df = pd.DataFrame(list(map(lambda x: 1 if x else 0, predictionsforrecall)), columns=['Numbers'])
    predictions_df.to_csv(prediction_file_name, index=False, header=False)

    AD1 = float(
        subprocess.run(
            ['./evaluation/evaluate', '-t', anomaly_ranges_file_name, prediction_file_name, '1', '1', 'one', 'flat', 'flat'], 
            stdout=subprocess.PIPE, 
            text=True)
            .stdout.split('\n')[1].split('=')[1].strip()
    )
    #  = ts_recall(anomalyranges, predictionsforrecall, alpha=1, cardinality="one", bias="flat")
    # AD2 = ts_recall(anomalyranges, predictionsforrecall, alpha=0, cardinality="one", bias="flat")
    AD2 = float(
        subprocess.run(
            ['./evaluation/evaluate', '-t', anomaly_ranges_file_name, prediction_file_name, '1', '0', 'one', 'flat', 'flat'], 
            stdout=subprocess.PIPE, 
            text=True)
            .stdout.split('\n')[1].split('=')[1].strip()
    )
    # AD3 = ts_recall(anomalyranges, predictionsforrecall, alpha=0, cardinality="one", bias="back")
    AD3 = float(
        subprocess.run(
            ['./evaluation/evaluate', '-t', anomaly_ranges_file_name, prediction_file_name, '1', '1', 'one', 'back', 'back'], 
            stdout=subprocess.PIPE, 
            text=True)
            .stdout.split('\n')[1].split('=')[1].strip()
    )

    os.remove(anomaly_ranges_file_name)
    os.remove(prediction_file_name)
    return AD1, AD2, AD3


def plotforevaluationNonFailure(timestyle,ignoredates,tempdatesofscores,episodethreshold,episodePred,countaxes,axes):
    if timestyle == "":
        for i in range(len(ignoredates)):
            if ignoredates[i][1] > tempdatesofscores[0] and ignoredates[i][1] < tempdatesofscores[-1]:
                pos1 = -1
                pos2 = -1
                for q in range(len(tempdatesofscores)):
                    if tempdatesofscores[q] > ignoredates[i][0]:
                        pos1 = q
                        break
                for q in range(pos1, len(tempdatesofscores)):
                    if tempdatesofscores[q] > ignoredates[i][1]:
                        pos2 = q
                        break
                # print(
                #     f"in episode {ignoredates[0].tz_localize(None)}-{ignoredates[-1].tz_localize(None)} we have {ignoredates[pos1]}-{ignoredates[pos2]} ")
                # ignoreperiod.extend(tempdatesofscores[pos1:pos2])
                axes[countaxes // 4][countaxes % 4].fill_between(
                    tempdatesofscores[pos1:pos2], max(max(episodethreshold), max(episodePred)),
                    min(episodePred),
                    color="grey",
                    alpha=0.3,
                    label="ignore")
    else:
        for i in range(len(ignoredates)):
            if ignoredates[i][1] > tempdatesofscores[0].tz_localize(None) and ignoredates[i][1] < \
                    tempdatesofscores[-1].tz_localize(None):
                pos1 = -1
                pos2 = -1
                for q in range(len(tempdatesofscores)):
                    if tempdatesofscores[q].tz_localize(None) > ignoredates[i][0]:
                        pos1 = q
                        break
                for q in range(pos1, len(tempdatesofscores)):
                    if tempdatesofscores[q].tz_localize(None) > ignoredates[i][1]:
                        pos2 = q
                        break
                # print(
                #     f"in episode {ignoredates[0].tz_localize(None)}-{ignoredates[-1].tz_localize(None)} we have {ignoredates[pos1]}-{ignoredates[pos2]} ")
                # ignoreperiod.extend(tempdatesofscores[pos1:pos2])
                axes[countaxes // 4][countaxes % 4].fill_between(
                    tempdatesofscores[pos1:pos2], max(max(episodethreshold), max(episodePred)),
                    min(episodePred),
                    color="grey",
                    alpha=0.3,
                    label="ignore")
    # ============================================================

    axes[countaxes // 4][countaxes % 4].plot(tempdatesofscores, episodePred, color="green", label="pb n")
    axes[countaxes // 4][countaxes % 4].plot(tempdatesofscores, episodethreshold, color="k", linestyle="--", label="th")
    # axes[countaxes//4][countaxes%4].legend()
    countaxes += 1
    return countaxes
def plotforevalurion(timestyle,ignoredates,tempdatesofscores,episodethreshold,episodePred,countaxes,axes,borderph, border1):
    if timestyle == "":
        for i in range(len(ignoredates)):
            if ignoredates[i][1] > tempdatesofscores[0] and ignoredates[i][1] < tempdatesofscores[-1]:
                pos1 = -1
                pos2 = -1
                for q in range(len(tempdatesofscores)):
                    if tempdatesofscores[q] > ignoredates[i][0]:
                        pos1 = q
                        break
                for q in range(pos1, len(tempdatesofscores)):
                    if tempdatesofscores[q] > ignoredates[i][1]:
                        pos2 = q
                        break
                # print(
                #     f"in episode {ignoredates[0].tz_localize(None)}-{ignoredates[-1].tz_localize(None)} we have {ignoredates[pos1]}-{ignoredates[pos2]} ")
                # ignoreperiod.extend(tempdatesofscores[pos1:pos2])
                axes[countaxes // 4][countaxes % 4].fill_between(
                    tempdatesofscores[pos1:pos2], max(max(episodethreshold), max(episodePred)),
                    min(episodePred),
                    color="grey",
                    alpha=0.3,
                    label="ignore")
    else:
        for i in range(len(ignoredates)):
            if ignoredates[i][1] > tempdatesofscores[0].tz_localize(None) and ignoredates[i][1] < \
                    tempdatesofscores[-1].tz_localize(None):
                pos1 = -1
                pos2 = -1
                for q in range(len(tempdatesofscores)):
                    if tempdatesofscores[q].tz_localize(None) > ignoredates[i][0]:
                        pos1 = q
                        break
                for q in range(pos1, len(tempdatesofscores)):
                    if tempdatesofscores[q].tz_localize(None) > ignoredates[i][1]:
                        pos2 = q
                        break
                # print(
                #     f"in episode {ignoredates[0].tz_localize(None)}-{ignoredates[-1].tz_localize(None)} we have {ignoredates[pos1]}-{ignoredates[pos2]} ")
                # ignoreperiod.extend(tempdatesofscores[pos1:pos2])
                axes[countaxes // 4][countaxes % 4].fill_between(
                    tempdatesofscores[pos1:pos2], max(max(episodethreshold), max(episodePred)),
                    min(episodePred),
                    color="grey",
                    alpha=0.3,
                    label="ignore")

    # ============================================================

    axes[countaxes // 4][countaxes % 4].plot(tempdatesofscores, episodePred, label="pb")
    axes[countaxes // 4][countaxes % 4].fill_between([tempdatesofscores[i] for i in range(borderph, border1)],
                                                     max(max(episodethreshold), max(episodePred)),
                                                     min(episodePred), where=[1 for i in range(borderph, border1)],
                                                     color="red",
                                                     alpha=0.3,
                                                     label="PH")
    axes[countaxes // 4][countaxes % 4].fill_between([tempdatesofscores[i] for i in range(border1, len(episodePred))],
                                                     max(max(episodethreshold), max(episodePred)),
                                                     min(episodePred),
                                                     where=[1 for i in range(border1, len(episodePred))],
                                                     color="grey",
                                                     alpha=0.3,
                                                     label="ignore")

    axes[countaxes // 4][countaxes % 4].plot(tempdatesofscores, episodethreshold, color="k", linestyle="--", label="th")

    # axes[countaxes//4][countaxes%4].legend()

    countaxes += 1
    return countaxes

def episodeMyPresicionTPFP(episodealarms,tempdatesofscores,PredictiveHorizon,leadtime,timestyle,timestylelead,ignoredates):
    border2=len(episodealarms)
    totaltp=0
    totalfp=0

    arraytoplot=[]
    toplotborders=[]


    border2date = tempdatesofscores[border2 - 1]

    border1 = border2 - 1
    if timestyle!="":
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - pd.Timedelta(leadtime,timestylelead):
                border1 = i -1
                if border1==-1:
                    border1=0
                break
        borderph=border1-1
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - pd.Timedelta(PredictiveHorizon,timestyle):
                borderph = i - 1
                if borderph==-1:
                    borderph=0
                break
        # print(f"len :{border2} , border 1: {border1}, borderph: {borderph}")
        positivepred = episodealarms[borderph:border1]
        negativepred = episodealarms[:borderph]
        negativepredDates = tempdatesofscores[:borderph]
        for value in positivepred:
            if value:
                totaltp += 1
        for value, valuedate in zip(negativepred, negativepredDates):
            if ignore(valuedate, tupleIngoredaes=ignoredates):
                if value:
                    totalfp += 1
        return totaltp, totalfp, borderph, border1

    else:
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - leadtime:
                border1 = i - 1
                break
        borderph = border1 - 1
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - PredictiveHorizon:
                borderph = i - 1
                break

        # print(f"len :{border2} , border 1: {border1}, borderph: {borderph}")
        positivepred = episodealarms[borderph:border1]
        negativepred = episodealarms[:borderph]
        negativepredDates = tempdatesofscores[:borderph]
        for value in positivepred:
            if value:
                totaltp += 1
        for value, valuedate in zip(negativepred, negativepredDates):
            if ingnorecounter(valuedate, tupleIngoredaes=ignoredates):
                if value:
                    totalfp += 1
        return totaltp, totalfp, borderph, border1


def _flatten(predictions):
    if isinstance(predictions[0], collections.abc.Sequence):
        temppreds=[]
        for episodepreds in predictions:
            temppreds.extend([pre for pre in episodepreds])
        predictions = temppreds
    return predictions

def format_only_predictions_dates(predictions,datesofscores):
    predictions=_flatten(predictions)
    datesofscores=_flatten(datesofscores)

    return predictions,datesofscores
def formulatedataForEvaluation(predictions,threshold,datesofscores,isfailure,maintenances):
    artificialindexes = []
    thresholdtempperepisode = []




    if isinstance(predictions[0], collections.abc.Sequence):
        temppreds = []
        maintenances = []
        if isinstance(threshold, collections.abc.Sequence) and len(threshold) == len(predictions):
            for episodepreds, thepisode in zip(predictions, threshold):
                if isinstance(thepisode, collections.abc.Sequence):
                    thresholdtempperepisode.extend(thepisode)
                else:
                    thresholdtempperepisode.extend([thepisode for i in range(len(episodepreds))])
                temppreds.extend([pre for pre in episodepreds])
                artificialindexes.extend([i + len(artificialindexes) for i in range(len(episodepreds))])
                maintenances.append(len(temppreds))
        else:
            for episodepreds in predictions:
                temppreds.extend([pre for pre in episodepreds])
                artificialindexes.extend([i + len(artificialindexes) for i in range(len(episodepreds))])
                maintenances.append(len(temppreds))
        predictions = temppreds
    else:
        artificialindexes = [i for i in range(len(predictions))]


    if len(datesofscores) == 0:
        datesofscores = artificialindexes
    elif isinstance(datesofscores[0], collections.abc.Sequence):
        temppreds = []
        for episodeindexesss in datesofscores:
            temppreds.extend(episodeindexesss)
        datesofscores = temppreds
    else:
        datesofscores=[dt for dt in datesofscores]

    if maintenances is not None and len(datesofscores) != 0:
        if maintenances[-1] != len(predictions) and type(maintenances[0]) == type(datesofscores[0]):
            newmain = []
            for maint in maintenances:
                if maint in datesofscores:
                    newmain.append(datesofscores.index(maint) + 1)
            if len(newmain) == len(maintenances):
                maintenances = newmain


    if maintenances is None:
        assert False, "When you pass a flatten array for predictions, maintenances must be assigned to cutoffs time/indexes"
    if maintenances[-1] != len(predictions):
        assert False, f"The maintenance indexes are not alligned with predictions length (last index of predictions should be the last element of maintenances) {maintenances[-1]} != {len(predictions)}"
    if len(predictions) != len(datesofscores):
        assert False, f"Inconsistency in the size of scores (predictions) and dates-indexes {len(predictions)} != {len(datesofscores)}"

    if len(isfailure) == 0:
        isfailure = [1 for m in maintenances]

    if isinstance(threshold, collections.abc.Sequence) and len(threshold) == len(maintenances):
        threshold = thresholdtempperepisode
    elif isinstance(threshold, collections.abc.Sequence) == False:
        temp = [threshold for i in predictions]
        threshold = temp

    assert len(predictions) == len(
        threshold), f"Inconsistency in the size of scores (predictions {len(predictions)}) and thresholds {len(threshold)}"

    return predictions,threshold,datesofscores,maintenances,isfailure

def calculatePHandLead(PH,lead):
    if len(PH.split(" "))<2:
        numbertime = int(PH.split(" ")[0])
        timestyle = ""
    else:
        scale = PH.split(" ")[1]
        acceptedvalues=["","days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"]
        if scale in acceptedvalues:
            numbertime = int(PH.split(" ")[0])

            timestyle=scale
        else:
            assert False,f"PH parameter must be in form \"number timescale\" e.g. \"8 hours\", where posible values for timescale are {acceptedvalues}"

    if len(lead.split(" "))<2:
        numbertimelead = int(lead.split(" ")[0])
        timestylelead = ""
    else:
        scale = lead.split(" ")[1]
        if scale in acceptedvalues:
            numbertimelead = int(lead.split(" ")[0])

            timestylelead = scale
        else:
            assert False,f"lead parameter must be in form \"number timescale\" e.g. \"8 hours\", where posible values for timescale are {acceptedvalues}"

    return numbertime,timestyle,numbertimelead,timestylelead

def ignore(valuedate,tupleIngoredaes):
    for tup in tupleIngoredaes:
        if valuedate.tz_localize(None)>tup[0] and valuedate.tz_localize(None)<tup[1]:
            return False
    return True
def ingnorecounter(valuedate,tupleIngoredaes):
    for tup in tupleIngoredaes:
        if valuedate>tup[0] and valuedate<tup[1]:
            return False
    return True




def breakIntoEpisodes(alarms,failuredates,thresholds,dates):
    isfailure=[]
    episodes=[]
    episodesthreshold=[]
    episodesdates=[]
    #dates=[pd.to_datetime(datedd) for datedd in dates]
    #failuredates=[pd.to_datetime(datedd) for datedd in failuredates]

    failuredates = [fdate for fdate in failuredates if fdate > dates[0]]

    # no failures
    if len(failuredates)==0 or len(dates)==0:
        if len(alarms)>0:
            isfailure.append(0)
            episodes.append(alarms)
            episodesthreshold.append(thresholds)
            episodesdates.append(dates)
        return isfailure,episodes,episodesdates,episodesthreshold

    failuredates = [fdate for fdate in failuredates if fdate > dates[0]]

    counter=0
    for fdate in failuredates:
        for i in range(counter,len(dates)):
            if dates[i]>fdate:
                if len(alarms[counter:i]) > 0:
                    isfailure.append(1)
                    episodes.append(alarms[counter:i])
                    episodesthreshold.append(thresholds[counter:i])
                    episodesdates.append(dates[counter:i])
                counter=i
                break
    if dates[-1]<failuredates[-1]:
        isfailure.append(1)
        episodes.append(alarms[counter:])
        episodesthreshold.append(thresholds[counter:])
        episodesdates.append(dates[counter:])
    elif counter<len(alarms):
        if len(alarms[counter:]) > 0:
            isfailure.append(0)
            episodes.append(alarms[counter:])
            episodesthreshold.append(thresholds[counter:])
            episodesdates.append(dates[counter:])
    return isfailure,episodes, episodesdates,episodesthreshold

def breakIntoEpisodesWithCodes(alarms,failuredates,failurecodes,thresholds,dates):
    isfailure=[]
    failuretype=[]
    episodes=[]
    episodesthreshold=[]
    episodesdates=[]
    #dates=[pd.to_datetime(datedd) for datedd in dates]
    #failuredates=[pd.to_datetime(datedd) for datedd in failuredates]
    failuredates = [fdate for fdate in failuredates if fdate > dates[0]]

    # no failures
    if len(failuredates)==0 or len(dates)==0:
        if len(alarms)>0:
            isfailure.append(0)
            failuretype.append("non failure")
            episodes.append(alarms)
            episodesthreshold.append(thresholds)
            episodesdates.append(dates)
        return isfailure,episodes,episodesdates,episodesthreshold,failuretype

    counter=0
    for fdate,ftype in zip(failuredates,failurecodes):
        for i in range(counter,len(dates)):
            if dates[i]>fdate:
                if len(alarms[counter:i]) > 0:
                    isfailure.append(1)
                    failuretype.append(ftype)
                    episodes.append(alarms[counter:i])
                    episodesthreshold.append(thresholds[counter:i])
                    episodesdates.append(dates[counter:i])
                counter=i
                break
    if dates[-1]<failuredates[-1]:
        isfailure.append(1)
        failuretype.append(failurecodes[-1])
        episodes.append(alarms[counter:])
        episodesthreshold.append(thresholds[counter:])
        episodesdates.append(dates[counter:])
    elif counter<len(alarms):
        if len(alarms[counter:]) > 0:
            isfailure.append(0)
            failuretype.append("non failure")
            episodes.append(alarms[counter:])
            episodesthreshold.append(thresholds[counter:])
            episodesdates.append(dates[counter:])
    return isfailure,episodes, episodesdates,episodesthreshold,failuretype




def extract_anomaly_ranges(maintenances,PHS_leads,isfailure,datesofscores):
    anomalyranges = [0 for i in range(maintenances[-1])]
    leadranges = [0 for i in range(maintenances[-1])]
    prelimit = 0
    counter = -1

    for maint, tupPHLEAD in zip(maintenances, PHS_leads):

        counter += 1
        if isfailure[counter] == 1:
            tempdatesofscores = datesofscores[prelimit:maint]
            borderph, border1,border_episode = Episode_Borders(tempdatesofscores, PredictiveHorizon=tupPHLEAD[1], leadtime=tupPHLEAD[3],
                                                               timestylelead=tupPHLEAD[4], timestyle=tupPHLEAD[2])
            if counter > 0:
                for i in range(borderph + prelimit, border1 + prelimit):
                    anomalyranges[i] = 1
            else:
                for i in range(borderph, border1):
                    anomalyranges[i] = 1

            if counter > 0:
                for i in range(border1 + prelimit, border_episode + prelimit):
                    leadranges[i] = 1
            else:
                for i in range(border1, border_episode):
                    leadranges[i] = 1

            prelimit = maint
    return anomalyranges,leadranges


def Episode_Borders(tempdatesofscores,PredictiveHorizon,leadtime,timestyle,timestylelead):
    border2 = len(tempdatesofscores)
    border2date = tempdatesofscores[border2 - 1]
    border1 = border2 - 1
    if timestyle != "":
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - pd.Timedelta(leadtime, timestylelead):
                border1 = i - 1
                if border1 == -1:
                    border1 = 0
                break
        borderph = border1 - 1
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - pd.Timedelta(PredictiveHorizon, timestyle):
                borderph = i - 1
                if borderph == -1:
                    borderph = 0
                break
        # print(f"len :{border2} , border 1: {border1}, borderph: {borderph}")
        return borderph, border1,border2
    else:
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - leadtime:
                border1 = i - 1
                break
        borderph = border1 - 1
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - PredictiveHorizon:
                borderph = i - 1
                break

        # print(f"len :{border2} , border 1: {border1}, borderph: {borderph}")
        return borderph, border1,border2


def _data_formulation(predictions,threshold,datesofscores,isfailure,maintenances,Failuretype,PH,lead):





    # format data to be in appropriate form for the evaluation, and check if conditions are true
    predictions, threshold, datesofscores, maintenances, isfailure = formulatedataForEvaluation(predictions, threshold,
                                                                                                datesofscores,
                                                                                                isfailure, maintenances)

    if Failuretype is None or Failuretype == [] or type(PH) is str:
        PH = [("type_all", PH)]
        lead = [("type_all", lead)]
        Failuretype = ["type_all" for i in maintenances]


    if len(maintenances) != len(Failuretype):
        assert False, "when using eval_multiPH the type of failure/maintenance, for each maintenance is required"
    if isinstance(PH, collections.abc.Sequence) == False:
        assert False, "when using eval_multiPH PH and lead parameter must be a list of tuples of form, (\"type name\",\"PH value\")"
    uniqueCodes = list(set(Failuretype))
    phcodes = [tupp[0] for tupp in PH]
    for cod in Failuretype:
        if cod not in phcodes:
            assert False, f"You must provide the ph for all different types in Failuretype, there are no info for {cod} in PH tuples"
    leadcodes = [tupp[0] for tupp in lead]
    for cod in Failuretype:
        if cod not in leadcodes:
            assert False, f"You must provide the lead for all different types in Failuretype, there are no info for {cod} in lead tuples"



    # calculate PH and lead
    PHS_leads = []
    for failuretype in Failuretype:
        posph = phcodes.index(failuretype)
        poslead = leadcodes.index(failuretype)
        tuplead = lead[poslead]
        tupPH = PH[posph]
        numbertime, timestyle, numbertimelead, timestylelead = calculatePHandLead(tupPH[1], tuplead[1])
        PHS_leads.append((failuretype, numbertime, timestyle, numbertimelead, timestylelead))
    return predictions, threshold, datesofscores, maintenances, isfailure,PHS_leads





# This method is used to perform PdM evaluation of Run-to-Failures examples.
# predictions: Either a flatted list of all predictions from all episodes or
#               list with a list of prediction for each of episodes
# datesofscores: Either a flatted list of all indexes (timestamps) from all episodes or
#                list with a list of  indexes (timestamps) for each of episodes
#                If it is empty list then aritificially indexes are gereated
# threshold: can be either a list of thresholds (equal size to all predictions), a list with size equal to number of episodes, a single number.
# maintenances: is used in case the predictions are passed as flatten array (default None)
#   list of ints which indicate the time of maintenance (the position in predictions where a new episode begins) or the end of the episode.
# isfailure: a binary array which is used in case we want to pass episode which end with no failure, and thus don't contribute
#   to recall calculation. For example isfailure=[1,1,0,1] indicates that the third episode end with no failure, while the others end with a failure.
#   default value is empty list which indicate that all episodes end with failure.
# PH: is the predictive horizon used for recall, can be set in time domain using one from accepted time spans:
#   ["days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"] using a space after the number e.g. PH="8 hours"
#   in case of single number then a counter related predictive horizon is used (e.g. PH="100" indicates that the last 100 values
#   are used for predictive horizon
# lead: represent the lead time (the time to ignore in last part of episode when we want to test the predict capabilities of algorithm)
#   same rules as the PH are applied.
# ignoredates: a list with tuples in form (begin,end) which indicate periods to be ignored in the calculation of recall and precision
#   begin and end values must be same type with datesofscores instances (pd.datetime or int)
# beta is used to calculate fbeta score deafult beta=1.
def pdm_eval_multi_PH(predictions, threshold, datesofscores=[], maintenances=None, isfailure=[],
                   PH=[("type 1", "100")], lead=[("type 1", "10")],Failuretype=[], plotThem=True, ignoredates=[], beta=1):

    predictions, threshold, datesofscores, maintenances, isfailure, PHS_leads=_data_formulation(predictions,threshold,datesofscores,isfailure,maintenances,Failuretype,PH,lead)


    anomalyranges,leadranges = extract_anomaly_ranges(maintenances,PHS_leads,isfailure,datesofscores)
    ignore_range=_ingore_dates_range_(datesofscores,ignoredates)

    recall, Precision, f1,FPR=calculate_AD_levels(anomalyranges, leadranges, predictions, ignore_range, threshold, beta)

    if plotThem:
        plt.plot(datesofscores, predictions)
        plt.plot(datesofscores, threshold)
        _inner_plot("red", anomalyranges, datesofscores, max(predictions))
        _inner_plot("grey", leadranges, datesofscores, max(predictions))
        plt.grid(True)
        plt.show()
    return recall, Precision, f1,FPR

def _ingore_dates_range_(datesofscores,ignoredates):
    ignore_range = [0 for i in range(len(datesofscores))]
    if len(ignoredates)==0:
        return ignore_range
    for i,date in enumerate(datesofscores):
        if ignore(date,ignoredates)==False:
            ignore_range[i]=1
    return ignore_range
def calculate_AD_levels(anomalyranges,leadranges,predictions,ignore_range,threshold,beta):
    if ignore_range is None:
        ignore_range=[0 for i in anomalyranges]
    totaltp = len([1 for an, pr, th, ld, ig in zip(anomalyranges, predictions, threshold, leadranges, ignore_range) if an == 1 and pr > th and ld==0 and ig==0])
    totalfp = len([1 for an, pr, th, ld, ig  in zip(anomalyranges, predictions, threshold, leadranges, ignore_range) if an == 0 and pr > th and ld==0 and ig==0])
    totalTN = len([1 for an, pr, th, ld, ig  in zip(anomalyranges, predictions, threshold, leadranges, ignore_range) if an == 0 and pr <= th and ld==0 and ig==0])


    if totalTN+totalfp>0:
        FPR=(totalfp/(totalTN+totalfp))
    else:
        FPR = 0
    predictionsinner=[1 if pr > th else 0 for  pr, th  in zip(predictions, threshold)]
    ### Calculate AD levels
    if sum(predictions) == 0:
        AD1 = 0
        AD2 = 0
        AD3 = 0
    else:
        AD1, AD2, AD3 = calculate_ts_recall(anomalyranges, predictionsinner)

        AD3 = AD3 * AD2

    ### Calculate Precision
    Precision = 0
    if totaltp + totalfp != 0:
        Precision = totaltp / (totaltp + totalfp)
    recall = [AD1, AD2, AD3]

    ### F ad scores
    f1 = []
    for rec in recall:
        if Precision + rec == 0:
            f1.append(0)
        else:
            F = ((1 + beta ** 2) * Precision * rec) / (beta ** 2 * Precision + rec)
            f1.append(F)
    return recall, Precision, f1,FPR





def AUCPR_anom(predictions,anomalyranges,leadranges=None,resolution=100,beta=1,slidingWindow_vus=0):
    """

    :param predictions: predictions (anomaly scores) in form of list of lists with predictions, or a single list of predictions
    :param anomalyranges: labels of 0 and 1 (has to be in the shame shape of predictions)
    :param leadranges: (default None) USed in case we want to ignore scores (considered only in precision calculation, and used in predictive maintenance scenarios)
    :param resolution: How many thresholds we should consider to calculate AUC PR.
    :param beta: F-beta parameter
    :param slidingWindow_vus: Sliding window for VUS calculation
    :return:  allresults -> lisf of shape [resolution,11] : for each revolution [AD1 f-beta,AD2 f-beta,AD3 f-beta,AD1 recall,AD2 recall,AD3 recall,precision,threshold,AUC AD1, AUC AD2, AUC AD3]
    results -> dictionary with VUS results:
    """

    predictions=_flatten(predictions)
    anomalyranges=_flatten(anomalyranges)
    unique,resolution,step,flatened_scores=_iterate_posible_thresholds(predictions, resolution)

    if leadranges is None:
        leadranges=[0 for i in anomalyranges]

    allresults,best_th=_calculate_AD_levels_AUC(resolution, unique, step,predictions, anomalyranges,
                                                                                          leadranges, beta,
                                                                                            None)
    results = _VUS_results(flatened_scores, anomalyranges, best_th, slidingWindow_vus)

    return allresults, results


def AUCPR_new(predictions, Failuretype=None, datesofscores=[], maintenances=None, isfailure=[], PH="100", lead="20",
              ignoredates=[], beta=1, resolution=100, slidingWindow_vus=0,plot_them=False):

    unique,resolution,step,flatened_scores=_iterate_posible_thresholds(predictions, resolution)

    threshold=[0 for qqq in predictions]



    predictions, threshold, datesofscores, maintenances, isfailure, PHS_leads = _data_formulation(predictions,
                                                                                                  threshold,
                                                                                                  datesofscores,
                                                                                                  isfailure,
                                                                                                  maintenances,
                                                                                                  Failuretype, PH,
                                                                                                  lead)
    anomalyranges, leadranges = extract_anomaly_ranges(maintenances, PHS_leads, isfailure, datesofscores)


    ignore_range = _ingore_dates_range_(datesofscores, ignoredates)

    allresults,best_th=_calculate_AD_levels_AUC(resolution, unique, step, predictions, anomalyranges, leadranges, beta, ignore_range)

    results=_VUS_results(flatened_scores, anomalyranges, best_th, slidingWindow_vus)

    if plot_them:
        plt.plot([i for i in range(len(predictions))],predictions)
        plt.plot([i for i in range(len(predictions))],[best_th for i in datesofscores])
        plt.fill_between([i for i in range(len(predictions))], min(predictions), max(predictions), where=anomalyranges, color="red",
                         alpha=0.3, label="PH")
        #plt.fill_between([i for i in range(predictions)], 0, max(predictions), where=leadranges, color="red",
        #                 alpha=0.3, label="PH")
        #_inner_plot("grey", leadranges,datesofscores, max(predictions))
        plt.grid(True)
        plt.legend()
        plt.show()
    return allresults, results

def _inner_plot(color,rangearay,datesofscores,maxvalue):

    change_indices = np.where(np.diff(rangearay) != 0)[0]
    if len(change_indices)%2==1:
        change_indices=np.append(change_indices,len(datesofscores))
    for i in range(0, len(change_indices), 2):
        plt.fill_between([datesofscores[q] for q in range(change_indices[i], min(change_indices[i+1]+1,len(datesofscores)))], 0, maxvalue, color=color,
                         alpha=0.3)







def _VUS_results(flatened_scores,anomalyranges,best_th,slidingWindow_vus):
    #### VUS RESULTS
    flatened_scores = np.array(flatened_scores)
    anomalyranges_for_vus = np.array(anomalyranges)
    scaler = MinMaxScaler(feature_range=(0, 1))
    score = scaler.fit_transform(flatened_scores.reshape(-1, 1)).ravel()
    results = get_metrics(score, anomalyranges_for_vus,
                          best_threshold_examined=scaler.transform(np.array([[best_th]])).ravel()[0],
                          slidingWindow=slidingWindow_vus)  # default metric='all'
    return results
def _AUC_from_results(allresults,tups_R_P1, tups_R_P2, tups_R_P3):
    allresultsforbestthreshold = allresults.copy()
    allresultsforbestthreshold.sort(key=lambda tup: tup[0], reverse=False)
    best_th = allresultsforbestthreshold[-1][-2]
    tups_R_P1 = sorted(tups_R_P1, key=lambda x: (x[0], -x[1]))
    # tups_R_P1.sort(key=lambda tup: tup[0], reverse=False)  # sorts in place
    tups_R_P2 = sorted(tups_R_P2, key=lambda x: (x[0], -x[1]))
    # tups_R_P2.sort(key=lambda tup: tup[0], reverse=False)  # sorts in place
    tups_R_P3 = sorted(tups_R_P3, key=lambda x: (x[0], -x[1]))
    # tups_R_P3.sort(key=lambda tup: tup[0], reverse=False)  # sorts in place

    recalls1 = [0] + [tup[0] for tup in tups_R_P1]
    recalls2 = [0] + [tup[0] for tup in tups_R_P2]
    recalls3 = [0] + [tup[0] for tup in tups_R_P3]

    press1 = [1] + [tup[1] for tup in tups_R_P1]
    press2 = [1] + [tup[1] for tup in tups_R_P2]
    press3 = [1] + [tup[1] for tup in tups_R_P3]

    # plt.subplot(121)
    # figtoplot=plt.figure(figsize=(28, 16))
    # ax = figtoplot.add_subplot()
    # ax.plot(recalls1,press1,"-o")
    # plt.plot(recalls2,press2)
    # plt.plot(press3,recalls3)

    if len(recalls1) == 1 or len(press1) == 1:
        AUC1 = 0.0
    else:
        AUC1 = sklearn.metrics.auc(recalls1, press1)

    if len(recalls2) == 1 or len(press2) == 1:
        AUC2 = 0.0
    else:
        AUC2 = sklearn.metrics.auc(recalls2, press2)

    if len(recalls3) == 1 or len(press3) == 1:
        AUC3 = 0.0
    else:
        AUC3 = sklearn.metrics.auc(recalls3, press3)

    for i in range(len(allresults)):
        allresults[i].append(AUC1)
        allresults[i].append(AUC2)
        allresults[i].append(AUC3)

    return allresults,best_th


def _calculate_AD_levels_AUC(resolution,unique,step,predictions,anomalyranges,leadranges,beta,ignore_range):
    allresults, tups_R_P1, tups_R_P2, tups_R_P3 = _calculate_AD_levels_for_all_thresholds(resolution, unique, step,
                                                                                          predictions, anomalyranges,
                                                                                          leadranges, beta,
                                                                                          ignore_range)
    allresults, best_th = _AUC_from_results(allresults, tups_R_P1, tups_R_P2, tups_R_P3)
    return allresults,best_th


def _calculate_AD_levels_for_all_thresholds(resolution,unique,step,predictions,anomalyranges,leadranges,beta,ignore_range):
    tups_R_P1 = []
    tups_R_P2 = []
    tups_R_P3 = []
    allresults = []

    for i in range(resolution + 2):
        examined_th = unique[min(i * step, len(unique) - 1)]
        threshold = [examined_th for predcsss in predictions]

        recall, Precision, f1, FPR = calculate_AD_levels(anomalyranges, leadranges, predictions, ignore_range,
                                                    threshold, beta)
        if Precision < 0.0000000000000001 and recall[0] < 0.0000000000000000001:
            continue

        tups_R_P1.append((recall[0], Precision))
        tups_R_P2.append((recall[1], Precision))
        tups_R_P3.append((recall[2], Precision))
        # All results
        allresults.append([f1[0], f1[1], f1[2], recall[0], recall[1], recall[2], Precision, examined_th,FPR])
    # allresults.append([0,0,0,1,1,1,0,min(unique)])
    allresults.append([0, 0, 0, 0, 0, 0, 1, max(unique),0])
    return allresults,tups_R_P1,tups_R_P2,tups_R_P3


def _iterate_posible_thresholds(predictions,resolution):
    predtemp = []
    if isinstance(predictions[0], collections.abc.Sequence):
        for predcs in predictions:
            predtemp.extend(predcs)
        flatened_scores = [kati for kati in predtemp]
        predtemp = list(set(predtemp))
    else:
        flatened_scores = [kati for kati in predictions]
        predtemp = list(set(predictions))
    predtemp.sort()
    unique = list(set(predtemp))
    unique.sort()
    resolution = min(resolution, max(1, len(unique)))
    step = int(len(unique) / resolution)
    return unique,resolution,step,flatened_scores






