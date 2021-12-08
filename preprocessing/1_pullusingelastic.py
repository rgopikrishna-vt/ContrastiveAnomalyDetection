import pandas as pd
import numpy as np
import json
import bz2
import pdb
import argparse
import pickle as pkl
import gzip
import elasticsearch as es
from datetime import datetime as dt
from datetime import timedelta,timezone
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_min_max_timestamp(esConn,index,field):
    """
        esConn: Elasticsearch connection object.
        index: Index to use.
        field: The field to aggregate on.
    """
    field="@timestamp"    
    # Get Max Timestamp In Index
    maxtstp_query = {
    "aggs": {
        "max_timestamp": {
            "max": {
            "field": field }
            }
        }
    }

    #Get Min Timestamp In Index
    mintstp_query = {
    "aggs": {
        "min_timestamp": {
            "min": {
            "field": field }
            }
        }
    }

    objs_max=esConn.search(index=index,body=json.dumps(maxtstp_query),size=0)
    objs_min=esConn.search(index=index,body=json.dumps(mintstp_query),size=0)

    maxtstp=objs_max['aggregations']['max_timestamp']['value']
    mintstp=objs_min['aggregations']['min_timestamp']['value']
    
    
    return maxtstp,mintstp


def main(esConn,index,outputdir,time_intervals):
    """
        @param esConn: Elasticsearch connection object.
        @param index: The name of the index.
        @param outputdir: Directory where .csv files are written.
        @param time_intervals: An array like object of times.
        @param min_frequency: The minimum frequency to be used.
        @param max_frequency: The maximum frequency to be used.
    """
    num_objects=list()
    binned_frequencies=dict()
    NUMOUTPUTFILES=200
    FILEID=1
    for i in range(len(time_intervals)-1):
        if i%2000==0:
            print("Start Time = {}, End Time = {}".format(dt.utcfromtimestamp(time_intervals[i]),
                                                          dt.utcfromtimestamp(time_intervals[i+1])))
            print("{} of {}".format(i ,len(time_intervals)-1))

#         query= {
#                 "query" : {
#                          "bool" : {
#                             "filter" : [
#                                 {
#                                     "range" : {
#                                         "@timestamp" : { "gte" :int(time_intervals[i])*1000,
#                                                   "lte" : int(time_intervals[i+1])*1000}
#                                     }
#                                 },
#                                 {
#                                      "range" : {
#                                         "signal.center_frequency" :{ "gte" : min_frequency, "lte" : max_frequency}
#                                     }
#                                 }
#                         ]
#                     }
#                 }
#             }
        
        query= {
                "query" : {
                         "bool" : {
                            "filter" : [
                                {
                                    "range" : {
                                        "@timestamp" : { "gte" :int(time_intervals[i])*1000,
                                                  "lte" : int(time_intervals[i+1])*1000}
                                    }
                                }
                        ]
                    }
                }
            }
        

        objs=esConn.search(index=index,body=json.dumps(query),size=10000)
        print("St. Time = {}, End Time ={} Num Objs = {}".format(dt.utcfromtimestamp(time_intervals[i]),dt.utcfromtimestamp(time_intervals[i+1]),len(objs['hits']['hits'])))
               
        num_objects.append(len(objs['hits']['hits']))
        df=pd.DataFrame([obj['_source'] for obj in objs['hits']['hits']])
        start_date = dt.strftime(dt.utcfromtimestamp(time_intervals[i]),"%Y-%m-%d-%H-%M-%S")
        end_date = dt.strftime(dt.utcfromtimestamp(time_intervals[i+1]),"%Y-%m-%d-%H-%M-%S")
        with bz2.BZ2File("{}/{}_{}.pkl.bz2".format(outputdir,start_date,end_date), 'w') as f:
            pkl.dump(objs['hits']['hits'],f)

        print("iter = {} of {} Num objects = {} at time = {}".format(i,len(time_intervals)-1,num_objects[-1],dt.utcfromtimestamp(time_intervals[i+1])))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--frequency_band",type=str,default='p25')
    args=parser.parse_args()


    TIMEDELTASECONDS = 180
    connection_string = ""
    INDEX="newomnisig-kop-*"
    
    
    outputdir="/home/rgopikrishna/data/newomnisig-kop-*/alldata_7Apr21to23May21"

    try:
        esConn = es.Elasticsearch([connection_string],verify_certs=False, ssl_show_warn=False)
    except Exception as e:
        print("ERROR connecting to Elastic: " + str(e))
        exit(1)


    """
           Sample :
              {'date': 1602256995114,
		 'hour_utc': 15,
		 'rssi': -0.9337654838369909,
		 'hour_local': 15,
		 'bandwidth': 24507.730268001556,
		 'energy_estimate': 149648.82566552862,
		 'confidence': 0.9999544024467468,
		 'frequency_upper_edge': 851994675.740134,
		 'day_of_week_utc': 'Friday',
		 'in_expected_band': 'Yes',
		 'description': 'P25',
		 'day_of_week_local': 'Friday',
		 'snr_estimate': 6.106188701648849,
		 'center_frequency': 851982421.875,
		 'mhz_band': 800,
		 'frequency_lower_edge': 851970168.009866,
		 'sample_count': 65536,
		 'global_index': 1256827655}

          Use mltest-base indices to baseline ML models as data is static.
          This data is on Tim's machine /opt/anomaly/data 
    """


    #Calculate Temporal Bins.
    #maxtstp,mintstp = get_min_max_timestamp(esConn,index=INDEX,field='date')
    maxtstp,mintstp = get_min_max_timestamp(esConn,index=INDEX,field='@timestamp')

    print("Min Tstp = {}, Max Tstp = {}".format(dt.utcfromtimestamp(int(mintstp)/1000),dt.utcfromtimestamp(int(maxtstp)/1000)))
    
    mintstp = dt(2021, 4, 7).timestamp()*1000
    maxtstp = dt(2021, 5, 23).timestamp()*1000

    print("Modified min and max times based on requirement:\nMin Tstp = {}, Max Tstp = {}".format(dt.utcfromtimestamp(int(mintstp)/1000),dt.utcfromtimestamp(int(maxtstp)/1000)))

    #Calculate Time Intervals.
    time_intervals = [mintstp/1000]
    while time_intervals[-1]<(maxtstp/1000 - TIMEDELTASECONDS):
        tstp = time_intervals[-1] + TIMEDELTASECONDS
        time_intervals.append(tstp)
    
    #Final condition which includes maxtstp if the final `time_interval` value is too far away from maxtstp.
    if time_intervals[-1]<(maxtstp/1000 - TIMEDELTASECONDS/2):
        time_intervals.append(maxtstp/1000)


    #time_intervals = np.linspace(mintstp,maxtstp,TIMEDELTASECONDS)
    print("Start = {}, \nEnd = {}, \nFinal Time = {}".format(dt.utcfromtimestamp(int(mintstp)/1000),
            dt.utcfromtimestamp((int(mintstp)/1000)+TIMEDELTASECONDS),dt.utcfromtimestamp(int(maxtstp)/1000)))

    main(esConn,INDEX,outputdir,time_intervals)
