#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Isabel Cenamor"
__copyright__ = "Copyright 2018, Portfolio Project Features"
__email__ = "icenamor@inf.uc3m.es"
import time
from datetime import datetime

# -----------------------------------------------------------------------------
## Class store the weka head
# -----------------------------------------------------------------------------
class Head:
    def __init__(self, list_head, planner_list):
        "store the Weka head"
        self.head = list_head
        self.planner_list = planner_list
        self.set_values(planner_list)

    def set_values(self, planner_list):
        tp = int(time.time())
        problem = (
            "@relation problem-to-evaluate"
            + str(datetime.now().strftime("%Y-%m-%d%H:%M:%S"))
            + "\n\n"
        )
        self.head.append(problem)
        self.head.append("@attribute domain string\n")
        self.head.append("@attribute task_name string\n")
        self.head.append("@attribute requirements numeric\n")
        self.head.append("@attribute types numeric\n")
        self.head.append("@attribute objects numeric\n")
        self.head.append("@attribute predicates numeric\n")
        self.head.append("@attribute functions numeric\n")
        self.head.append("@attribute init numeric\n")
        self.head.append("@attribute goal numeric\n")
        self.head.append("@attribute actions numeric\n")
        self.head.append("@attribute axioms numeric\n")
        self.head.append("@attribute use_min_cost_metric numeric\n")
        self.head.append("@attribute generated_rules numeric\n")
        self.head.append("@attribute relevant_atoms numeric\n")
        self.head.append("@attribute auxiliary_atoms numeric\n")
        self.head.append("@attribute final_queue_length numeric\n")
        self.head.append("@attribute total_queue_pushes numeric\n")
        self.head.append("@attribute implied_effects_removed numeric\n")
        self.head.append("@attribute effect_conditions_simplified numeric\n")
        self.head.append("@attribute implied_preconditions_added numeric\n")
        self.head.append("@attribute translator_variables numeric\n")
        self.head.append("@attribute translator_derived_variables numeric\n")
        self.head.append("@attribute translator_facts numeric\n")
        self.head.append("@attribute translator_mutex_groups numeric\n")
        self.head.append("@attribute translator_total_mutex_groups_size numeric\n")
        self.head.append("@attribute translator_operators numeric\n")
        self.head.append("@attribute translator_task_size numeric\n")
        self.head.append("@attribute numberVariablesCG numeric\n")
        self.head.append("@attribute highLevelVariablesCG numeric\n")
        self.head.append("@attribute totalEdgesCG numeric\n")
        self.head.append("@attribute totalWeightCG numeric\n")
        self.head.append("@attribute veRatio numeric\n")
        self.head.append("@attribute weRatio numeric\n")
        self.head.append("@attribute wvRatio numeric\n")
        self.head.append("@attribute hvRatio numeric\n")
        self.head.append("@attribute inputEdgeCGMax numeric\n")
        self.head.append("@attribute inputEdgeCGAvg numeric\n")
        self.head.append("@attribute inputEdgeCGStd numeric\n")
        self.head.append("@attribute outputEdgeCGMax numeric\n")
        self.head.append("@attribute outputEdgeCGAvg numeric\n")
        self.head.append("@attribute outputEdgeCGStd numeric\n")
        self.head.append("@attribute inputWeightCGMax numeric\n")
        self.head.append("@attribute inputWeightCGAvg numeric\n")
        self.head.append("@attribute inputWeightCGStd numeric\n")
        self.head.append("@attribute outputWeightCGMax numeric\n")
        self.head.append("@attribute outputWeightCGAvg numeric\n")
        self.head.append("@attribute outputWeightCGStd numeric\n")
        self.head.append("@attribute inputEdgeHVMax numeric\n")
        self.head.append("@attribute inputEdgeHVAvg numeric\n")
        self.head.append("@attribute inputEdgeHVStd numeric\n")
        self.head.append("@attribute outputEdgeHVMax numeric\n")
        self.head.append("@attribute outputEdgeHVAvg numeric\n")
        self.head.append("@attribute outputEdgeHVStd numeric\n")
        self.head.append("@attribute inputWeightHVMax numeric\n")
        self.head.append("@attribute inputWeightHVAvg numeric\n")
        self.head.append("@attribute inputWeightHVStd numeric\n")
        self.head.append("@attribute outputWeightHVMax numeric\n")
        self.head.append("@attribute outputWeightHVAvg numeric\n")
        self.head.append("@attribute outputWeightHVStd numeric\n")
        self.head.append("@attribute numberVariablesDTG numeric\n")
        self.head.append("@attribute totalEdgesDTG numeric\n")
        self.head.append("@attribute totalWeigthDTG numeric\n")
        self.head.append("@attribute edVaRatioDTG numeric\n")
        self.head.append("@attribute weEdRatioDTG numeric\n")
        self.head.append("@attribute weVaRatioDTG numeric\n")
        self.head.append("@attribute inputEdgeDTGMax numeric\n")
        self.head.append("@attribute inputEdgeDTGAvg numeric\n")
        self.head.append("@attribute inputEdgeDTGStd numeric\n")
        self.head.append("@attribute outputEdgeDTGMax numeric\n")
        self.head.append("@attribute outputEdgeDTGAvg numeric\n")
        self.head.append("@attribute outputEdgeDTGStd numeric\n")
        self.head.append("@attribute inputWeightDTGMax numeric\n")
        self.head.append("@attribute inputWeightDTGAvg numeric\n")
        self.head.append("@attribute inputWeightDTGStd numeric\n")
        self.head.append("@attribute outputWeightDTGMax numeric\n")
        self.head.append("@attribute outputWeightDTGAvg numeric\n")
        self.head.append("@attribute outputWeightDTGStd numeric\n")
        self.head.append("@attribute no numeric\n")
        self.head.append("@attribute file string\n")
        self.head.append("@attribute num_relevant_facts numeric\n")
        self.head.append("@attribute num_actions numeric\n")
        self.head.append("@attribute h_max numeric\n")
        self.head.append("@attribute h_ff numeric\n")
        self.head.append("@attribute h_ff_ratio numeric\n")
        self.head.append("@attribute rp_fact_balance_min numeric\n")
        self.head.append("@attribute rp_fact_balance_avg numeric\n")
        self.head.append("@attribute rp_fact_balance_var numeric\n")
        self.head.append("@attribute rp_goal_balance_min numeric\n")
        self.head.append("@attribute rp_goal_balance_avg numeric\n")
        self.head.append("@attribute rp_goal_balance_var numeric\n")

        self.head.append("@attribute balance_Ratio numeric\n")
        self.head.append("@attribute Unbalance_Ratio numeric\n")
        self.head.append("@attribute Balance_Distorsion numeric\n")

        self.head.append("@attribute Additive numeric\n")
        self.head.append("@attribute Blind numeric\n")
        self.head.append("@attribute Causal_graph numeric\n")
        self.head.append("@attribute Context-enhanced_additive numeric\n")
        self.head.append("@attribute FF numeric\n")
        self.head.append("@attribute Goal_count numeric\n")
        self.head.append("@attribute Landmark_count numeric\n")
        self.head.append("@attribute Landmark-cut numeric\n")
        self.head.append("@attribute Max numeric\n")

        ##landMark graph

        ## features<<landmarks<<","<<numberEdges<<","<<numberEdges/double(landmarks)<<",";
        ##  features<<numberFather<<","<<numberChildren<< ","<<nodosbetween<<",";
        ##features<<avginput<<","<<maxinput<< ","<<stdinput<<","<<avgoutput<<","<<maxoutput<<","<<stdoutput<<endl;

        self.head.append("@attribute n_landmarks numeric\n")
        self.head.append("@attribute numberEdges numeric\n")
        self.head.append("@attribute edVaratioLand numeric\n")
        self.head.append("@attribute numberFatherNodes numeric\n")
        self.head.append("@attribute numberChildrenNodes numeric\n")
        self.head.append("@attribute nodosbetween numeric\n")

        self.head.append("@attribute avginput numeric\n")
        self.head.append("@attribute maxinput numeric\n")
        self.head.append("@attribute stdinput numeric\n")

        self.head.append("@attribute avgoutput numeric\n")
        self.head.append("@attribute maxoutput numeric\n")
        self.head.append("@attribute stdoutput numeric\n")

        self.head.append("@attribute blackVariables numeric\n")
        self.head.append("@attribute blackRootVariables numeric\n")

        self.head.append("@attribute redBlackVariables numeric\n")
        self.head.append("@attribute allpairsvaluesconnected numeric\n")
        self.head.append("@attribute allvaluesconnectedgoal numeric\n")

        self.head.append("@attribute blackstronglpyarents numeric\n")
        self.head.append("@attribute Maximaleffectsblack numeric\n")
        self.head.append("@attribute Hred-black numeric\n")
        # self.head.append("@attribute planner {tamer, enhsp, fast-downward, lpg}\n")
        res = "@attribute planner {"
        for i, p in enumerate(planner_list):
            if i == 0:
                res += str(p)
            else:
                res += ", " + str(p)
        res += "}\n"
        self.head.append(res)

        self.head.append("@attribute class {True,False}\n\n")

        self.head.append("@data\n\n")
