#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Isabel Cenamor"
__copyright__ = "Copyright 2018, Portfolio Project Features"
__email__ = "icenamor@inf.uc3m.es"

import sys
import string
import os
from up_ibacop.utils.models.head import Head


Ntranslate = 26
Npreprocess = 49
Nheuristic = 8
Nlandmark = 11
NredBlack = 7
NBalance = 15
##translateFile --> translate
##features.arff --> preprocess
##initfeature-info.txt --> ff-learner
##tmp_results --> heuristics
def readFile(name, datos):
    # print(name)
    fd = open(name, "r")
    linea = fd.readline()
    while linea != "":
        datos.append(linea)
        linea = fd.readline()
    return datos


def segurityNumberFeature(listFeatures, number):
    # print "wsegurity"
    aux = listFeatures.count(",")
    # print aux, number
    while aux < number:
        listFeatures = "?," + listFeatures
        aux = listFeatures.count(",")
    return listFeatures


def writeFile(name, data, head, res_planner):
    fd = open(name, "w")
    for i in head.head:
        fd.write(i)
    line = ""
    for i in data:
        line = line + i
    # use this print to see the features
    # print(line)
    for res in res_planner:
        entry = line + "," + res + "\n"
        fd.write(entry)
    # entry = line + ",tamer, True\n"
    # fd.write(entry)
    # entry = line + ",enhsp, True\n"
    # fd.write(entry)
    # entry = line + ",pyperplan, True\n"
    # fd.write(entry)
    # entry = line + ",lgp, True\n"
    # fd.write(entry)
    fd.close()


def join(translate, preprocess, fflearner, heuristics, landmarks, redblack, union):
    if len(translate) > 0:
        # print "translate", Ntranslate, translate[0].count(',')
        if translate[0].count(",") < Ntranslate:
            translate[0] = segurityNumberFeature(translate[0], Ntranslate)
        union = union + translate[0]
    else:
        # print("There is not translate")
        entry_translate = "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?"
        union = union + entry_translate
    union = union + ","
    if len(preprocess) > 0:
        # print "preprocess", Npreprocess, preprocess[0].count(',')
        if preprocess[0].count(",") < Npreprocess:
            preprocess[0] = segurityNumberFeature(preprocess[0], Npreprocess)
        union = union + preprocess[0][: len(preprocess[0]) - 1]
    else:
        # print("There is not preprocess")
        entry_translate = "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?"
        union = union + entry_translate
    union = union + ","
    if len(fflearner) > 0:
        # print NBalance, fflearner[0].count(',')
        if fflearner[0].count(",") < NBalance:
            fflearner[0] = segurityNumberFeature(fflearner[0], NBalance)
        union = union + fflearner[0][: len(fflearner[0]) - 1]
    else:
        # print("There is not fflearner")
        entry_translate = "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?"
        union = union + entry_translate
    union = union + ","
    if len(heuristics) > 0:
        # print "heuristics", Nheuristic, heuristics[0].count(',')
        if heuristics[0].count(",") < Nheuristic:
            heuristics[0] = segurityNumberFeature(heuristics[0], Nheuristic)
        union = union + heuristics[0][: len(heuristics[0]) - 1]
    else:
        # print("There is not heuristics")
        entry_translate = "?,?,?,?,?,?,?,?,?"
        union = union + entry_translate
    union = union + ","
    if len(landmarks) > 0:
        # print "landmarks", Nlandmark, landmarks[0].count(',')
        if landmarks[0].count(",") < Nlandmark:
            landmarks[0] = segurityNumberFeature(landmarks[0], Nlandmarks)
        union = union + landmarks[0][: len(landmarks[0]) - 1]
    else:
        # print("There is not landmarks")
        entry_translate = "?,?,?,?,?,?,?,?,?,?,?,?"
        union = union + entry_translate
    union = union + ","
    if len(redblack) > 0:
        # print "redblack", NredBlack, redblack[0].count(',')
        if redblack[0].count(",") < NredBlack:
            redblack[0] = segurityNumberFeature(redblack[0], NredBlack)
            # print "estoy aqui"
        union = union + redblack[0][: len(redblack[0]) - 1]
    else:
        # print("There is not red-black")
        entry_translate = "?,?,?,?,?,?,?,?"
        union = union + entry_translate
    ##union = union.replace("-nan", "?")
    ##union = union.replace("nan", "?")
    ##union = union.replace("-inf", "?")
    ##union = union.replace("inf", "?")
    return union


def create_globals(route, res_planner, plannerList):

    translate = []
    preprocess = []
    fflearner = []
    heuristics = []
    landmarks = []
    redblack = []
    union_final = ""

    try:
        translate = readFile(route + "/translateFile", translate)  ## translateFile
    except:
        # print("No file in translate")
        translate = ["?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?"]
    try:
        preprocess = readFile(route + "/features.arff", preprocess)  ## features.arff
    except:
        # print("No file in preprocess")
        preprocess = [
            "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?\n"
        ]
    try:
        fflearner = readFile(route + "/initfeature-info.txt", fflearner)
    except:
        # print("No file in fflearner")
        fflearner = ["?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?\n"]
    try:
        ##the route is wrong
        landmarks = readFile(route + "/landmark.arff", landmarks)
    except:
        # print("No file in landmarks")
        landmarks = ["?,?,?,?,?,?,?,?,?,?,?,?\n"]
    try:
        ##the route is wrong
        redblack = readFile(route + "/red-black", redblack)
    except:
        # print("No file in red-black")
        redblack = ["?,?,?,?,?,?,?,?\n"]
    try:
        heuristics = readFile(route + "/tmp_results", heuristics)
    except:
        # print("No file in heuristics")
        heuristics = ["?,?,?,?,?,?,?,?,?\n"]
    try:
        union_final = join(
            translate,
            preprocess,
            fflearner,
            heuristics,
            landmarks,
            redblack,
            union_final,
        )
    except:
        # print("General error")
        union_final = "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?"
    head = Head([], plannerList)

    union_final = union_final.replace("-nan", "?")
    union_final = union_final.replace("nan", "?")
    union_final = union_final.replace("-inf", "?")
    union_final = union_final.replace("inf", "?")
    union_final = union_final.replace("-2147483647", "-1000")
    union_final = union_final.replace("2147483647", "1000")
    writeFile(route + "/global_features.arff", union_final, head, res_planner)
