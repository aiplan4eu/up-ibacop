#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import string
import os


class Instance:
    "store Instance"

    def __init__(self, a, b):
        self.problem = a
        self.domain = b

    def clear_line(self, string, instance):
        i = string.find(",")
        instance.domain = string[:i]
        i = i + 1
        instance.problem = int(string[i : len(string) - 1])
        return instance

    def __str__(self):
        name = "problem: " + str(self.problem) + " domain: " + str(self.domain)
        return name


## inst#     actual  predicted error prediction (planner)
class Result:
    "store the Result"

    def __init__(self, a, b, c, d, f):
        self.inst = a
        self.actual = b
        self.predicted = c
        self.error = d
        self.planner = f

    def __str__(self):
        name = (
            "inst: "
            + str(self.inst)
            + " actual: "
            + str(self.actual)
            + " predicted: "
            + str(self.predicted)
            + " error: "
            + str(self.error)
            + " planner: "
            + str(self.planner)
        )
        return name

    def clear_actual_value(self, string):
        ##1:?
        i = string.find(":")
        return string[i + 1 :]

    def clear_predicted_value(self, string):
        ##1:true
        i = string.find(":")
        return string[i + 1 :]

    def clear_planner_value(self, string):
        ##(planner)
        i = string.find("(")
        return string[i + 1 : len(string) - 1]

    ## 1        1:?     1:true       1 (arvand)
    ##5     1:True    2:False   +   0.778 (fdss-2)
    def split_line(self, string):
        string = string.strip()  ##trim
        space = string.find(" ")
        string = string.replace("+", " ")
        tab = string.find("\t")
        if space > 0 and tab == -1:
            inst = int(string[:space])
            string = string[(space + 1) :]
            string = string.strip()
            space = string.find(" ")
            actual = self.clear_actual_value(string[:space])
            string = string[(space + 1) :]
            string = string.strip()
            space = string.find(" ")
            predicted = self.clear_predicted_value(string[:space])
            string = string[(space + 1) :]
            string = string.strip()
            space = string.find(" ")
            error = float(string[:space])
            if predicted.lower().find("false") >= 0:
                error = 1 - error

            string = string[(space + 1) :]
            string = string.strip()
            planner = self.clear_planner_value(string)
            result = Result(inst, actual, predicted, error, planner)
        else:
            print("Default case")
        self = result
        return self
