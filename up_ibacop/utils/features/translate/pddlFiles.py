#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__      = "Isabel Cenamor"
__copyright__   = "Copyright 2013, Portfolio Project -- Features translate"
__email__ = "icenamor@inf.uc3m.es"

##Create file with features
class PddlFile(object):
	def __init__(self, domain_name, task_name, requirements, types, objects, predicates, functions, init, goal, actions, axioms, use_metric):
		self.domain_name = domain_name
		self.task_name = task_name
		self.requirements = requirements
		self.types = types
		self.objects = objects
		self.predicates = predicates
		self.functions = functions
		self.init = init
		self.goal = goal
		self.actions = actions
		self.axioms = axioms
		self.use_min_cost_metric = use_metric
		
		##TODO translate. pddl_to_sas(task) instantiate.explore.build.model.compute_model(prog)
		##the variable go in this way
		self.generated_rules = -1
		self.relevant_atoms = -1
		self.auxiliary_atoms  = -1
		self.final_queue_length  = -1
		self.total_queue_pushes = -1 
		## simplify.apply_to_task. simplify.apply_to_operators(self.operators)
		self.operators_removed = -1 ##NO
		self.propositions_removed = -1 ##NO
		
		##invariant_finder.find_invariants(task)
		self.initial_candidates = -1 ##NO
		##TODO
		## translate.py
		self.implied_effects_removed = -1
		self.effect_conditions_simplified = -1
		self.implied_preconditions_added = -1
		## translate.py
		self.translator_variables = -1
		self.translator_derived_variables =  -1
		self.translator_facts =  -1
		self.translator_mutex_groups =  -1
		self.translator_total_mutex_groups_size =  -1
		self.translator_operators = -1
		self.translator_task_size =  -1
		
		
	def printObject(self, name):
		f = open(name, "w")
		values = str(self.domain_name) + "," + str(self.task_name) + "," +  str(self.requirements) + "," + \
		str(self.types) + "," + str(self.objects) + "," +  str(self.predicates) + "," + \
		str(self.functions) + "," + str(self.init) + "," +  str(self.goal) + "," + \
		str(self.actions) + "," + str(self.axioms) + "," +  str(self.use_min_cost_metric) + "," +\
		str(self.generated_rules) + "," + str(self.relevant_atoms) + "," + str(self.auxiliary_atoms ) + "," + \
		str(self.final_queue_length ) + "," + str(self.total_queue_pushes) + "," + \
		str(self.implied_effects_removed) + "," + str(self.effect_conditions_simplified) + "," +\
		str(self.implied_preconditions_added) + "," + \
		str(self.translator_variables) + "," + str(self.translator_derived_variables) + "," + \
		str(self.translator_facts) + "," + \
		str(self.translator_mutex_groups) + "," +\
		str(self.translator_total_mutex_groups_size) + "," +\
		str(self.translator_operators) + "," + str(self.translator_task_size)
		f.write(values)
		##str(self.operators_removed) + \
		##"," + str(self.propositions_removed ) + "," + \
		##str(self.initial_candidates) + "," + \
		f.close()
	
	def printObjectSimply(self, name):
		f = open(name, "w")
		values = str(self.types) + "," + str(self.objects) +"," + \
		str(self.functions) + "," +  str(self.goal) + "," + \
		str(self.auxiliary_atoms ) + "," + \
		str(self.implied_effects_removed) +\
		str(self.translator_facts) + "," + \
		str(self.translator_total_mutex_groups_size)
		f.write(values)
		f.close()
	
	def passMinCost(self, value):
		if(bool(value)):
			self.use_min_cost_metric = 1
		else:
			self.use_min_cost_metric = 0
			
	def dump(self):
		values = str(self.domain_name) + "," + str(self.task_name) + "," +  str(self.requirements) + "," + \
		str(self.types) + "," + str(self.objects) + "," +  str(self.predicates) + "," + \
		str(self.functions) + "," + str(self.init) + "," +  str(self.goal) + "," + \
		str(self.actions) + "," + str(self.axioms) + "," +  str(self.use_min_cost_metric) + "," +\
		str(self.generated_rules) + "," + str(self.relevant_atoms) + "," + str(self.auxiliary_atoms ) + "," + \
		str(self.final_queue_length ) + "," + str(self.total_queue_pushes) + "," + str(self.operators_removed) + \
		"," + str(self.propositions_removed ) + "," + \
		str(self.initial_candidates) + "," + \
		str(self.implied_effects_removed) + "," + str(self.effect_conditions_simplified) + "," +\
		str(self.implied_preconditions_added) + "," + \
		str(self.translator_variables) + "," + str(self.translator_derived_variables) + "," + \
		str(self.translator_facts) + "," + \
		str(self.translator_mutex_groups) + "," +\
		str(self.translator_total_mutex_groups_size) + "," +\
		str(self.translator_operators) + "," + str(self.translator_task_size)
		print(values)
		
	def dumpSimply(self):
		values = str(self.types) + "," + str(self.objects) +"," + \
		str(self.functions) + "," +  str(self.goal) + "," + \
		str(self.auxiliary_atoms ) + "," + \
		str(self.implied_effects_removed) +\
		str(self.translator_facts) + "," + \
		str(self.translator_total_mutex_groups_size)
		print(values)
