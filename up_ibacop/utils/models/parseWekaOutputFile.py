#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import string
import os
from up_ibacop.utils.models.Result import Result
from up_ibacop.utils.models.Result import Instance
from operator import itemgetter, attrgetter

STRATEGY = 4


def readFile(data, name):
    fd = open(name, "r")
    data = fd.readlines()
    fd.close()
    return data


def clear_data(data):
    data2 = []
    i = 0
    start = 0
    while i < len(data):
        if data[i].find("#") < 0 and start == 0:
            i = i + 1
        elif data[i].find("#") > 0 and start == 0:
            i = i + 1
            start = 1
        else:
            start = 1
        if start == 1:
            if len(data[i]) > 1:
                data2.append(data[i])
        i = i + 1
    data = data2
    return data


def sorted_results(data, sortedData):

    positive_list = []
    negative_list = []
    for i in data:
        if i.predicted.lower().find("false") >= 0:
            negative_list.append(i)
        else:
            positive_list.append(i)

    positive_list = sorted(positive_list, key=lambda result: result.error)
    positive_list = reversed(positive_list)
    negative_list = sorted(negative_list, key=lambda result: result.error)

    negative_list = reversed(negative_list)  ##Cambiar parte 7-jul debate Tomas
    for value in positive_list:
        sortedData.append(value)
    for value in negative_list:

        sortedData.append(value)
    return sortedData


def writeFile(sortedData, name, numberPlanner):
    fd = open(name, "w")
    i = 0
    while i < numberPlanner and len(sortedData) >= numberPlanner:
        planner = sortedData[i].planner
        fd.write(planner + "\n")
        i = i + 1
    fd.close()


def split_problems(data, listData):
    n = 0
    aux = 1
    for i in data:
        # print("*****", aux, len(listData))
        # print(i)
        if aux < 4:
            listData[n].append(i)
            aux = aux + 1
        else:
            listData[n].append(i)
            n = n + 1
            aux = 1
    return listData


def parseOutputFile(outputModel, listPlanner):
    data = []
    data = readFile(data, outputModel)
    data = clear_data(data)
    results = []
    for i in data:
        result = Result(0, "", "", 0.0, "")
        result = result.split_line(i)
        results.append(result)
    ## from more than one problem
    listData = []
    # print("results", len(results))
    for i in range(int(len(results) / 3)):
        listData.append([])
    listData = split_problems(results, listData)
    list_aux = []
    for i in listData:
        for j in i:
            list_aux.append(j)
    sortedData = []
    sortedData = sorted_results(list_aux, sortedData)

    # if(len(sys.argv) == 4):
    # 	writeFile(sortedData, sys.argv[2], int(sys.argv[3]))
    # else:
    writeFile(sortedData, listPlanner, STRATEGY)
