import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_all_data(list_train, list_test, context_list_dfs):
    dates_all=[]
    failures=[]
    contextdata= None
    alldftest = None
    for dftrain, dftest, dfcont in zip(list_train, list_test, context_list_dfs):
        dates_all.append([dtt for dtt in dftest.index])
        if alldftest is None:
            alldftest = dftest.copy()
            contextdata = dfcont.loc[dftest.index[0]:].copy()
        else:
            alldftest = pd.concat([alldftest, dftest])
            contextdata = pd.concat([contextdata, dfcont.loc[dftest.index[0]:].copy()])

        failures.append(dftest.index[-1])

    return alldftest,contextdata,dates_all,failures

def split_df_with_failures(df,failure_dates):
    df_list=[]
    pre_date=df.index[0]
    for fail_d in failure_dates:
        df_list.append(df.loc[pre_date:fail_d].copy())
        pre_date=fail_d
    df_list.append(df.loc[pre_date:].copy())
    return df_list

def split_df_with_failures_isfaile(df,failure_dates):
    df_list=[]
    isfailure=[]
    pre_date=df.index[0]
    for fail_d in failure_dates:
        df_list.append(df.loc[pre_date:fail_d].copy())
        pre_date=fail_d
        isfailure.append(1)
    df_list.append(df.loc[pre_date:].copy())
    isfailure.append(0)
    return df_list,isfailure

def generate_auto_train_test(df,period_or_count="100"):
    if len(period_or_count.split(" "))<2:
        numbertime = int(period_or_count.split(" ")[0])
        timestyle = ""
    else:
        scale = period_or_count.split(" ")[1]
        acceptedvalues=["","days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"]
        if scale in acceptedvalues:
            numbertime = int(period_or_count.split(" ")[0])
            timestyle=scale

    if timestyle=="":
        index_pos=df.index[numbertime]
    else:
        index_pos=df.index[0]+ pd.Timedelta(numbertime, timestyle)

    traindf=df.loc[:index_pos]
    testdf=df.loc[index_pos:]
    return traindf,testdf

def returnseperated(dictdata,module,type,isf):


    dfs = dictdata["data"]
    if len(dfs[0].index) == 0:
        lastind = dfs[1].index[-1]
    elif len(dfs[1].index) == 0:
        lastind = dfs[0].index[-1]
    else:
        lastind = max(dfs[0].index[-1], dfs[1].index[-1])

    maintenances = dictdata["maintenances"]
    if isf==0:
        maintvalues = maintenances.values
        maintvalues = np.vstack([maintvalues, ["eval", f"{module}_{type}"]])
        maintenances = pd.DataFrame(data=maintvalues, columns=maintenances.columns,
                                    index=[dtind for dtind in maintenances.index] + [lastind])
    events = dictdata["events"]

    sourceB = dictdata["sources"][0]
    sourceC = dictdata["sources"][1]

    failuretimes = dictdata["failures_info"][0]
    failuretimes = [dt.tz_localize(None) for dt in failuretimes]

    # failurecodes = dictdata["failures_info"][1]
    # failuresources = dictdata["failures_info"][2]
    #
    # eventsofint = dictdata["event_to_plot"]

    if "A" in sourceB:
        keepdf=dfs[0]
    else:
        keepdf=dfs[1]

    return keepdf,maintenances,events,failuretimes


def philips_semi_supervised(period_or_count="100"):
    dfs = []
    for i in range(8):
        df = pd.read_csv(f"m2_episode_{i}.csv", index_col=0, header=0)
        df.index = pd.to_datetime(df.index)
        dfs.append(df)

    typesdf = pd.read_csv("types.csv", index_col=0, header=0)
    names = [nam for nam in typesdf["name"]]
    types = [nam for nam in typesdf["types"]]

    dfffailure = pd.read_csv("files_end_with_failure.csv", index_col=0, header=0)
    isfailure = [isf for isf in dfffailure["end_with_fail"].values]

    event_lists = []
    for name in names:
        dfev = pd.read_csv(f"{name}.csv")
        lista = [dtt for dtt in pd.to_datetime(dfev["Timestamp"]).values]
        lista.sort()
        event_lists.append(lista)
    print(names)
    print(event_lists)
    print(isfailure)
    print(types)

    list_train = []
    list_test = []
    for i in range(len(dfs)):
        traindf, testdf = generate_auto_train_test(dfs[i], period_or_count)
        list_train.append(traindf)
        list_test.append(testdf)
    return list_train, list_test, names, event_lists, types, isfailure


def get_type(name):
    if "weld" in name:
        typeve = "configuration"
    elif "speed" in name or "-change" in name:
        typeve = "configuration"
    elif "8" in name or "47" in name or "44" in name or "42" in name or "31" in name or "51" in name or "43" in name or "80" in name:
        typeve = "configuration"
    else:
        typeve = "isolated"
    return typeve



def getSignleMachineData(machine_id=1):
    dftelemetry = pd.read_csv("./Data/azurepdm/PdM_telemetry.csv", header=0)
    dfmainentance = pd.read_csv("./Data/azurepdm/PdM_maint.csv", header=0)
    dferrors = pd.read_csv("./Data/azurepdm/PdM_errors.csv", header=0)
    dffailures = pd.read_csv("./Data/azurepdm/PdM_failures.csv", header=0)
    dfmachines = pd.read_csv("./Data/azurepdm/PdM_machines.csv", header=0)

    dfmainentance = dfmainentance[dfmainentance["machineID"] == machine_id]
    dferrors = dferrors[dferrors["machineID"] == machine_id]
    dffailures = dffailures[dffailures["machineID"] == machine_id]
    dftelemetry = dftelemetry[dftelemetry["machineID"] == machine_id]

    dftelemetry["datetime"] = pd.to_datetime(dftelemetry["datetime"])
    dferrors["datetime"] = pd.to_datetime(dferrors["datetime"])
    dffailures["datetime"] = pd.to_datetime(dffailures["datetime"])
    dfmainentance["datetime"] = pd.to_datetime(dfmainentance["datetime"])


    dffailures["code"]=[f"f_{comp}" for comp in dffailures["failure"]]
    dferrors["code"]=dferrors["errorID"]
    dfmainentance["code"]=[f"m_{comp}" for comp in dfmainentance["comp"]]
    source=f"{machine_id}"

    dftelemetry.index=dftelemetry['datetime']
    dftelemetry=dftelemetry.drop(["datetime","machineID"],axis=1)
    dferrors.index=dferrors['datetime']
    dferrors = dferrors.drop(["datetime"], axis=1)
    dfmainentance.index=dfmainentance['datetime']
    dfmainentance = dfmainentance.drop(["datetime"], axis=1)
    dffailures.index=dffailures['datetime']
    dffailures = dffailures.drop(["datetime"], axis=1)

    return dftelemetry,dferrors,dffailures,dfmainentance,source




def AzureDataOneSource(source=1):

    dftelemetry,dferrors,dffailures,dfmainentance,source=getSignleMachineData(machine_id=source)
    contextdf = {}
    sumerrors=0
    summaint=0
    for errorname in dferrors["errorID"].unique():
        indexes = list(dferrors[dferrors['errorID'] == errorname].index)
        for ind in indexes:
            if ind not in dftelemetry.index:
                for tind in dftelemetry.index:
                    if tind>ind:
                        indexes.append(tind)
                        break
        errors = [1 if time in indexes else 0 for time in dftelemetry.index]
        ferrors=[]
        count=0
        for i in range(len(errors)):
            if errors[i]>0:
                count+=1
            ferrors.append(count)

        contextdf[errorname]=ferrors
        sumerrors+=sum(errors)

    for maint in dfmainentance["comp"].unique():
        indexes = list(dfmainentance[dfmainentance['comp'] == maint].index)
        for ind in indexes:
            if ind not in dftelemetry.index:
                for tind in dftelemetry.index:
                    if tind > ind:
                        indexes.append(tind)
                        break
        maintenance = [1 if time in indexes else 0 for time in dftelemetry.index]
        fmaintenance = []
        count = 0
        for i in range(len(maintenance)):
            if maintenance[i] > 0:
                count += 1
            fmaintenance.append(count)

        contextdf[maint]=fmaintenance
        summaint+=sum(maintenance)


    dfcontext=pd.DataFrame(contextdf)
    dfcontext.index=dftelemetry.index

    failures= [dt for dt in dffailures.index]


    # dfcontext.plot()
    # for fail in failures:
    #     plt.axvline(fail)
    # plt.show()
    return dftelemetry,dfcontext,failures


def AzureData():
    all_dfs=[]
    all_context=[]
    all_isfailure=[]
    all_sources=[]
    for source in range(1,101):
        dftelemetry,dfcontext,failures=AzureDataOneSource(source=source)
        dfs,isfailure=split_df_with_failures_isfaile(dftelemetry, failures)
        dfs_context,_=split_df_with_failures_isfaile(dfcontext, failures)

        all_dfs.extend(dfs)
        all_context.extend(dfs_context)
        all_isfailure.extend(isfailure)
        all_sources.extend([source for ep in dfs])
    return all_dfs,all_context,all_isfailure,all_sources



def Azure_generate_train_test(list_of_df,isfailure,context_list_dfs,all_sources, period_or_count=f"200 hours"):
    # PH 96 hours profile 200
    list_train = []
    list_test = []
    new_isfailure = []
    context_list_dfs_new = []
    new_sources = []
    for i in range(len(list_of_df)):
        traindf, testdf = generate_auto_train_test(list_of_df[i], period_or_count)
        if testdf.shape[0]<2:
            continue
        new_isfailure.append(isfailure[i])
        list_train.append(traindf)
        list_test.append(testdf)
        context_list_dfs_new.append(context_list_dfs[i])
        new_sources.append(all_sources[i])
    return list_train, list_test,context_list_dfs_new,new_isfailure,new_sources


def AzureDataOneSource_list(source=1):

    dftelemetry,dferrors,dffailures,dfmainentance,source=getSignleMachineData(machine_id=source)

    event_list=[]
    names=[]
    types=[]

    failures= [dt for dt in dffailures.index]
    event_list.append(failures)
    names.append("failures")
    types.append("configuration")
    for errorname in dferrors["errorID"].unique():
        indexes = list(dferrors[dferrors['errorID'] == errorname].index)
        event_list.append(indexes)
        names.append(f"error_{errorname}")
        types.append("configuration")
    for errorname in dfmainentance["comp"].unique():
        indexes = list(dfmainentance[dfmainentance['comp'] == errorname].index)
        event_list.append(indexes)
        names.append(f"comp_{errorname}")
        types.append("configuration")

    # dfcontext.plot()
    # for fail in failures:
    #     plt.axvline(fail)
    # plt.show()
    return dftelemetry,event_list,names,types,"failures"
