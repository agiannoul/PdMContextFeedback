import pickle
import statistics
import random
import time

import numpy as np
import pandas as pd
from PdmContext.utils.simulate_stream import simulate_from_df, simulate_stream
from PdmContext.ContextGeneration import ContextGenerator, ContextGeneratorBatch
from PdmContext.utils.causal_discovery_functions import calculate_with_pc
from PdmContext.utils.dbconnector import SQLiteHandler
from PdmContext.Pipelines import ContextAndDatabase
from PdmContext.utils.structure import Context, Eventpoint
from PdmContext.utils.distances import distance_cc
import kshape.core as kcore
from pathlib import Path

from tqdm import tqdm


class PruneFP():
    """
    This class handles the prune of potentially false positives alarms.

    Leverages pre-calculated contexts by using ContextGeneratorBatch instead of ContextGenerator for faster experiments.

    self.database_name is the path to pre-calculated contexts
    """
    def __init__(self,data,contextdata,scores,threshold_score,times,evaluations,threshold_similarity=0.5,alpha=0.5,context_horizon="8 hours",username="Metro",consider_FP="36 hours",add_raw_to_context=True,checkincrease=False):
        self.threshold_score=threshold_score
        self.consider_FP=consider_FP
        self.scores=scores
        self.times=times
        self.threshold_similarity=threshold_similarity
        self.evaluations=evaluations
        self.alpha=alpha
        self.Fps = []
        self.checkincrease=checkincrease
        if username is not None:
            Dataset_name = username.split("_")[0]
            self.database_name=f"./Databases/{Dataset_name}/ch_{context_horizon.replace(' ','_')}_alpha_0.5_{username}.pickle"
        else:
            self.database_name=None
        df=pd.DataFrame({"scores":scores})
        df.index=data.index

        if add_raw_to_context:
            for col in data.columns:
                df[col] = data[col]
        event_types=[]
        for col in contextdata.columns:
            df[col]=contextdata[col]

        type_of_series={}
        for col in df.columns:
            if col in contextdata.columns:
                type_of_series[col]="configuration"

        traget_name = "scores"

        self.con_gen=ContextGeneratorBatch(df,traget_name,type_of_series,context_horizon=context_horizon,
                                      Causalityfunct=calculate_with_pc,debug=False,file_path=self.database_name)


    def get_context(self,timestamp):
        if timestamp in self.con_gen.contexts.keys():
            return self.con_gen.contexts[timestamp]
        else:
            return self.con_gen.generate_context(timestamp)

    def prune_scores(self):

        self.false_positives=[-1 for i in range(len(self.scores))]
        predict=[]
        next_evaluation_pos=0



        #for qi in tqdm(range(len(self.scores))):
        for qi in range(len(self.scores)):
            sc=self.scores[qi]
            dtt=self.times[qi]
            if sc <= self.threshold_score:
                predict.append(0)
            else:
                tempcontext=self.get_context(dtt)
                # if len([1 for key in  tempcontext.CD.keys() if "comp" in key or "error" in key ])>1:
                #     tempcontext.plot()
                if self.similar_to_fp(tempcontext):
                    predict.append(0)
                else:
                    predict.append(1)
                # perform pruning

            if next_evaluation_pos<len(self.evaluations):
                if dtt>=self.evaluations[next_evaluation_pos]:
                    self.create_fps(self.evaluations[next_evaluation_pos])
                    next_evaluation_pos+=1



        return predict
    def similar_to_fp(self,context: Context):
        for con in self.Fps:
            similarity,similarities=my_distance(con, context, a=self.alpha, verbose=False)
            #print(similarities)
            if similarity>self.threshold_similarity:
                return True
        return False
    def getFalsepositives(self):
        return [con.timestamp for con in self.Fps]
    def create_fps(self,date):
        tempdate=date-pd.Timedelta(int(self.consider_FP.split(" ")[0]), self.consider_FP.split(" ")[1])

        for i in range(len(self.false_positives)):
            if self.false_positives[i]!=-1:
                continue
            if self.times[i]>date:
                break
            if self.times[i]<tempdate and self.scores[i]>self.threshold_score:
                self.false_positives[i]=1
            else:
                self.false_positives[i]=0
        self.Fps = []
        for dtt, fp in zip(self.times, self.false_positives):
            if fp == 1:
                temp_context = self.get_context(dtt)
                if self.checkincrease:
                    for edge, charac in zip(temp_context.CR["edges"], temp_context.CR["characterization"]):
                        if edge[1] == "scores" and charac == "increase":
                            self.Fps.append(temp_context)
                            break
                else:
                    self.Fps.append(temp_context)
        self.Fps=[self.get_context(dtt) for dtt,fp in zip(self.times,self.false_positives) if fp==1]
        



class PruneFPstream():
    """
    This class is used to apply Feedback calculate in Philips Dataset.

    Leverages stored distance (and stores new calculated distances between contexts) for faster experiments.

    self.database_name: is the path to Calculated contexts (Sqlite3 database)
    self.known_distance_name: is the path to a pickle file, containing calculated distance using 'my_distance' function
    in form: key=(timestamp1,timestamp2): value = (SBD,JACCARD)

    """
    def __init__(self, df,  scores, threshold_score, times, evaluations,contain_raw_data=False,
                 contextdata_list=[],types=[],names=[],
                 threshold_similarity=0.5, alpha=0.5, context_horizon="8 hours",
                 username="Philips", consider_FP="8 hours",
                 savedistances=False,checkincrease=True):
        self.checkincrease=checkincrease
        self.threshold_score = threshold_score
        self.consider_FP = consider_FP
        self.scores = scores
        self.times = times
        self.savedistances = savedistances

        event_triple=[]
        for lista,ltype,name in zip(contextdata_list,types,names):
            event_triple.append((name,lista,ltype))

        continuous_triple=[]
        if contain_raw_data:
            for col in df.columns:
                continuous_triple.append((col,df[col].values,df.index))

        continuous_triple.append(("scores", scores, df.index))

        self.threshold_similarity = threshold_similarity
        self.evaluations = evaluations
        self.alpha = alpha

        self.Fps = []
        self.contextlist= {}

        self.database_name = f"./Databases/stream/ch_{context_horizon.replace(' ', '_')}_{username}.db"
        self.known_distance_name = f"./Databases/stream/distances/ch_{context_horizon.replace(' ', '_')}_alpha_{self.alpha}_{username}.pickle"
        self.traget_name = "scores"
        self.dont_save=False
        my_file = Path(self.database_name)
        if my_file.is_file() and username is not None:
            self.load_context()
            self.known_distance=self._load_distances()
        else:
            self.known_distance={}
            self.traget_name = "scores"
            sizeall=sum([len(lista[1]) for lista in continuous_triple]) + sum([len(lista) for lista in contextdata_list])


            stream = simulate_stream(continuous_triple, event_triple,[], self.traget_name)

            con_gen = ContextGenerator(target=self.traget_name, context_horizon=context_horizon, Causalityfunct=calculate_with_pc)
            source = "press"

            database = SQLiteHandler(db_name=self.database_name)
            contextpipeline = ContextAndDatabase(context_generator_object=con_gen, databaseStore_object=database)

            #print(sizeall)
            #counter=0
            allrecords=[record for record in stream]
            # for record in tqdm(allrecords):
            #     #counter+=1
            #     #if counter%(sizeall//100)==0:
            #     #    print(counter/sizeall)
            #     contextpipeline.collect_data(timestamp=record["timestamp"], source=source, name=record["name"],
            #                                   type=record["type"], value=record["value"])
            # tempcontextlist=contextpipeline.Contexter.contexts
            # self.contextlist = {}
            # for cont in tempcontextlist:
            #     self.contextlist[cont.timestamp] = cont
            tempcontextlist=[]
            for record in tqdm(allrecords):
                #counter+=1
                #if counter%(sizeall//100)==0:
                #    print(counter/sizeall)
                if record["name"] == self.traget_name and record["value"] > self.threshold_score:
                    context_obj=contextpipeline.collect_data(timestamp=record["timestamp"], source=source, name=record["name"],
                                              type=record["type"], value=record["value"])
                    tempcontextlist.append(context_obj)
                else:
                    eventpoint = Eventpoint(code=record["name"], source=source, timestamp=record["timestamp"], details=record["value"], type=record["type"])
                    contextpipeline.Contexter.add_to_buffer(eventpoint)


            self.contextlist = {}
            for cont in tempcontextlist:
                self.contextlist[cont.timestamp] = cont


    def load_context(self):
        database = SQLiteHandler(db_name=self.database_name)
        traget_name = "scores"
        tempcontextlist = database.get_all_context_by_target(traget_name)
        self.contextlist={}
        for cont in tempcontextlist:
            self.contextlist[cont.timestamp]=cont
    def prune_scores(self):

        self.false_positives = [-1 for i in range(len(self.scores))]
        predict = []
        next_evaluation_pos = 0
        counter=0
        for sc, dtt in zip(self.scores, self.times):
            # counter+=1
            # if counter%(len(self.scores)//100)==0:
            #     print(counter/len(self.scores))
            if sc <= self.threshold_score:
                predict.append(0)
            else:
                if dtt not in self.contextlist.keys():
                    predict.append(0)
                elif self.similar_to_fp(self.contextlist[dtt]):
                    predict.append(0)
                else:
                    predict.append(1)
                # perform pruning

            if next_evaluation_pos < len(self.evaluations):
                if dtt >= self.evaluations[next_evaluation_pos]:
                    self.create_fps(self.evaluations[next_evaluation_pos])
                    next_evaluation_pos += 1

        return predict

    def create_fps(self, date):
        tempdate = date - pd.Timedelta(int(self.consider_FP.split(" ")[0]), self.consider_FP.split(" ")[1])

        for i in range(len(self.false_positives)):
            if self.false_positives[i] != -1:
                continue
            if self.times[i] > date:
                break
            if self.times[i] < tempdate and self.scores[i] > self.threshold_score:
                self.false_positives[i] = 1
            else:
                self.false_positives[i] = 0

        self.Fps = []
        for dtt, fp in zip(self.times, self.false_positives):
            if fp == 1:
                if dtt not in self.contextlist.keys():
                    continue
                temp_context=self.contextlist[dtt]
                if self.checkincrease:
                    for edge, charac in zip(temp_context.CR["edges"],temp_context.CR["characterization"]):
                        if edge[1]==self.traget_name and charac=="increase":
                            self.Fps.append(temp_context)
                            break
                else:
                    self.Fps.append(temp_context)
        #random.shuffle(self.Fps)
        #print(len(self.Fps))
    def similar_to_fp(self, context: Context):
        for con in self.Fps:
            tup=(con.timestamp,context.timestamp)
            if tup in self.known_distance.keys():
                similarities=self.known_distance[tup]
            else:
                _,similarities = my_distance(con, context, a=self.alpha, verbose=False)
                self.known_distance[tup]=similarities
            if similarities[1] is None or similarities[0] is None:
                ok="ok"
            similarity=self.alpha*similarities[0]+(1-self.alpha)*similarities[1]
            if similarity > self.threshold_similarity:
                return True
        return False

    def _save_distances(self):
        if self.savedistances:
            with open(self.known_distance_name, 'wb') as f:
                pickle.dump(self.known_distance, f)

    def _load_distances(self):
        my_file = Path(self.known_distance_name)
        if my_file.is_file():
            with open(self.known_distance_name, 'rb') as f:
                return pickle.load(f)
            self.dont_save=True
        else:
            return {}

    def __del__(self):
        if self.dont_save==False:
            self._save_distances()

def my_distance(context1: Context, context2: Context, a, verbose=False):
    """
    Calculation of similarity between two Context objects based on two quantities:
        1) The first quantity is based on the sbd distance
            We calculate the minimum (average) sbd between all common series in the CD of contexts, from all possible shifts.
            The shifts apply to all series each time.
            Each time we use the last n values (where n is the size of the shorter series)
            Which is also weighted from the ratio of common values.
        2) Jaccard similarity of the edges in the CR (if we ignore the direction)

    **context1**: A context object

    **context2**: A context object

    **a**: the weight of SBD similarity

    **verbose**:

    **return**: a similarity between 0 and 1
    """
    if len(context1.CD.keys())<1:
        return 0,(0,0)
    if len(context2.CD.keys())<1:
        return 0,(0,0)

    #print("========================================")
    #step1=time.time()
    b = 1 - a
    common_values = []
    uncommon_values = []
    for key in context1.CD.keys():
        if key in context2.CD.keys() and context1.CD[key] is not None and context2.CD[key] is not None:
            common_values.append(key)
        else:
            uncommon_values.append(key)
    for key in context2.CD.keys():
        if key not in context1.CD.keys():
            uncommon_values.append(key)
    #step2 = time.time()
    #print(f"{step2-step1} : common_names")
    context1series=[]
    context2series=[]
    #if len(common_values) > 1 and a > 0.0000000001 and len(context2.CD[common_values[0]]) > 5 and len(context1.CD[common_values[0]]) > 5:
    if len(common_values)==0:
        return 0,(0,0)
    if a > 0.0000000001 and len(context2.CD[common_values[0]]) > 5 and len(context1.CD[common_values[0]]) > 5:
        All_common_cc = []
        sizee = min(len(context1.CD[common_values[0]]), len(context2.CD[common_values[0]]))
        for key in common_values:
            #step11 = time.time()
            All_common_cc.append(key)
            firtsseries = context1.CD[key][-sizee:]
            secondseries = context2.CD[key][-sizee:]
            # step21 = time.time()
            # print(f"{step21 - step11} : cut")
            #firtsseries = _z_norm_np(firtsseries)
            #secondseries = _z_norm_np(secondseries)
            # step31 = time.time()
            # print(f"{step31 - step21} : normalize")
            context1series.append(firtsseries)
            context2series.append(secondseries)
        #step3 = time.time()
        #print(f"{step3 - step2} : normalize and gather")
        in_cc_m = np.max(kcore._ncc_c_3dim([np.array(context1series), np.array(context2series)]))
        #step4 = time.time()
        #print(f"{step4 - step3} : SBD")
        cc_m = in_cc_m * len(All_common_cc) / (len(All_common_cc) + len(uncommon_values))
        if verbose:
            print(f"Common cc_m = {in_cc_m}")
            print(f"uncommon_values: {len(uncommon_values)}")
            print(f"Final cc_m = {cc_m}")
    else:
        cc_m = 0
    # cc_m ε [-1,1] -> [0,1]

    # check common causes-characterizations:
    similarity=calculate_jaccard(a, context1, context2)

    if similarity is None:
        return cc_m, (cc_m, 0)
    else:
        return a * cc_m + b * similarity,(cc_m, similarity)







def calculate_jaccard(a,context1,context2):
    b=1-a
    if b > 0.000000001:
        # check common causes-characterizations:
        common = 0

        edges1 = ignore_order(context1)
        edges2 = ignore_order(context2)

        for edge in edges1:
            for edge2 in edges2:
                if edge[0] == edge2[0] and edge[1] == edge2[1]:
                    common += 1

        if (len(edges1) + len(edges2) - common) > 0:
            if common == 0:
                jaccard = 0
            else:
                jaccard = common / (len(edges1) + len(edges2) - common)
            similarity = jaccard
        # there are no samples Jaccard(empty,empty) = ? , in that case we use only first part
        else:
            if a < 0.0000001:
                similarity = 1
            else:
                similarity = None
    else:
        similarity = 0
    return similarity


def ignore_order(context1: Context):
    edges1 = []

    for edge in context1.CR['edges']:
        if edge[0] > edge[1]:
            potential = (edge[0], edge[1])
        else:
            potential = (edge[1], edge[0])
        if potential not in edges1:
            edges1.append(potential)
    return edges1

def _z_norm(series):
    if min(series) != max(series):
        ms1 = statistics.mean(series)
        ss1 = statistics.stdev(series)
        series = [(s1 - ms1) / ss1 for s1 in series]
    else:
        series = [0 for i in range(len(series))]
    return series

def _z_norm_np(series):
    if np.min(series) != np.max(series):
        ms1 = np.mean(series)
        ss1 = np.std(series)
        series = (series - ms1) / ss1
    else:
        series = np.zeros_like(series)
    return series
