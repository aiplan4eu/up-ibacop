
/**
** @author Isabel Cenamor <icenamor@inf.uc3m.es> 2013
**/
#ifndef SIMPLYCG_H
#define SIMPLYCG_H

#include <iostream>
#include <vector>
/**
** @author Isabel Cenamor <icenamor@inf.uc3m.es> 2013
**/
using namespace std;

class simplyCG {
	int source;
	int sucessor;
	int target;
	int value;
public:
	simplyCG(int s, int succ, int t, int va);
	simplyCG();
	int getSource();
	int getSucessor();
	int getTarget();
	int getValue();
};
#endif

#ifndef SIMPLYDTG_H
#define SIMPLYSTG_H

class simplyDTG {
	int variable;
	int numTrans;
	int source;
	int target;
	int conditions;
public:
	simplyDTG(int v, int nT, int s, int t, int con);
	simplyDTG();
	int getVariable();
	int getNumTrans();
	int getTarget();
	int getSource();
	int getConditions();
	
	
};
#endif

#ifndef STORE_H
#define STORE_H

class Store {
	vector<simplyCG> causal_graph;
	vector<simplyDTG> domain_transion_graph;
	vector<int> goals;
public:
	void addSimplyCG(simplyCG causal);
	void addSimplyDTG(simplyDTG dtg);
	void addGoals(int value);
	vector<simplyCG> getCausalGraph();
	vector<simplyDTG> getDomainTransitionGraph();
	vector<int> getGoals();
	int getVariablesCG();
	int getTotalWeightCG();
	void setCausalGraph(vector<simplyCG> simplycg);
	void setDomainTransitionGraph(vector<simplyDTG> simplydtg);
	void setGoals(vector<int> goal);
	void printFile(ofstream &outfile);
};
#endif

#ifndef VALUES_H
#define VALUES_H

class Values {
public:
	int var;
	double inputEdgeDTGMax;
	double inputEdgeDTGAvg;
	double inputEdgeDTGStd;
	double outputEdgeDTGMax;
	double outputEdgeDTGAvg;
	double outputEdgeDTGStd;
	
	double inputWeightDTGMax;
	double inputWeightDTGAvg;
	double inputWeightDTGStd;
	double outputWeightDTGMax;
	double outputWeightDTGAvg;
	double outputWeightDTGStd;
	void print();
};
#endif
