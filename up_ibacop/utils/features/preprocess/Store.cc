/**
** @author Isabel Cenamor <icenamor@inf.uc3m.es> 2013
**/

#include "Store.h"

#include "helper_functions.h"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cassert>

using namespace std;

simplyCG::simplyCG(){
	source = -1;
	sucessor = -1;
	target = -1;
	value = -1;
}

simplyCG::simplyCG(int s, int succ, int t, int va){
	source = s;
	sucessor = succ;
	target = t;
	value = va;
}

int simplyCG::getSource(){
	return source;
}
int simplyCG::getSucessor(){
	return sucessor;
}
int simplyCG::getTarget(){
	return target;
}
int simplyCG::getValue(){
	return value;
}


simplyDTG::simplyDTG(){
	source = -1;
	target = -1;
	variable = -1;
	numTrans = -1;
	conditions = -1;
}

simplyDTG::simplyDTG(int v, int nT, int s, int t, int con){
	source = s;
	target = t;
	variable = v;
	numTrans = nT;
	conditions = con;
}

int simplyDTG::getVariable(){
	return variable;
}
int simplyDTG::getNumTrans(){
	return numTrans;
}
int simplyDTG::getTarget(){
	return target;
}
int simplyDTG::getSource(){
	return source;
}
int simplyDTG::getConditions(){
	return conditions;
}


void Store::addSimplyCG(simplyCG causal){
	causal_graph.push_back(causal);
}
void Store::addSimplyDTG(simplyDTG dtg){
	domain_transion_graph.push_back(dtg);
}
void Store::addGoals(int value){
	goals.push_back(value);
}
vector<simplyCG> Store::getCausalGraph(){
	return causal_graph;
}
vector<simplyDTG> Store::getDomainTransitionGraph(){
	return domain_transion_graph;
}
vector<int> Store::getGoals(){
	return goals;
}
void Store::setCausalGraph(vector<simplyCG> simplycg){
	causal_graph = simplycg;
}

void Store::setDomainTransitionGraph(vector<simplyDTG> simplydtg){
	domain_transion_graph = simplydtg;
}
void Store::setGoals(vector<int> goal){
	goals = goal;
}

void Store::printFile(ofstream &outfile){

	outfile.open("featuresGraph", ios::out);
	outfile<<"CG"<<endl;
	outfile<<getVariablesCG()<<endl;
	for (int i = 0; i < causal_graph.size(); i++) {
		outfile<<causal_graph[i].getSource()<<", "<<causal_graph[i].getSucessor()<<", "<<causal_graph[i].getTarget()<<", "<<causal_graph[i].getValue()<<endl;
	}
	outfile<<"DTG"<<endl;
	for (int i = 0; i < domain_transion_graph.size(); i++) {
		outfile<<domain_transion_graph[i].getVariable()<<", "<<domain_transion_graph[i].getNumTrans()<<", "<<domain_transion_graph[i].getSource()<<", "<<domain_transion_graph[i].getTarget()<<", "<<domain_transion_graph[i].getConditions()<<endl;
	}
	outfile<<"Goal"<<endl;
	for (int i = 0; i < goals.size(); i++) {
		outfile<<getGoals()[i]<<endl;
	}
	outfile.close();
	
}

int Store::getVariablesCG(){
	int max_variable = 0;
	vector<simplyCG> auxiliar = getCausalGraph();
	for (int i = 0; i < auxiliar.size();i++){
		if(auxiliar[i].getSource() > max_variable) {
			 max_variable = auxiliar[i].getSource();
		}
		if(auxiliar[i]. getTarget() > max_variable) {
			 max_variable = auxiliar[i]. getTarget();
		}
	}
	max_variable = max_variable + 1;
	
	return max_variable;
}


int Store::getTotalWeightCG(){
	int weight = 0;
	vector<simplyCG> auxiliar = getCausalGraph();
	for (int i = 0; i < auxiliar.size();i++){
		weight =  weight + auxiliar[i].getValue();
	}
	return weight;
}

void Values::print(){
	cout<<var<<"  ,  "<<inputEdgeDTGMax<<"  ,  "<<inputEdgeDTGAvg<<"  ,  "<<inputEdgeDTGStd<<"  ,  "<<outputEdgeDTGMax<<"  ,  "<<outputEdgeDTGAvg<<"  ,  "<<outputEdgeDTGStd<<" , "<<inputWeightDTGMax<<"  ,  "<<inputWeightDTGAvg<<"  ,  "<<inputWeightDTGStd<<"  ,  "<<outputWeightDTGMax<<"  ,  "<<outputWeightDTGAvg<<"  ,  "<<outputWeightDTGStd<<endl;

}
