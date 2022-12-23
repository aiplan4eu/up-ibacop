#ifndef FEATURE_H
#define FEATURE_H

#include <iostream>
#include <vector>
/**
** @author Isabel Cenamor <icenamor@inf.uc3m.es> 2013
**/


class simplyCG;
class Store;
class simplyDTG;
class Values;


using namespace std;

class Features{

	//Causal Gragh
	int numberVariablesCG;
	int highLevelVariablesCG;
	int totalEdgesCG;
	int totalWeightCG;
	double veRatio;
	double weRatio;
	double wvRatio;
	double hvRatio;
	
	double inputEdgeCGMax;
	double inputEdgeCGAvg;
	double inputEdgeCGStd;
	double outputEdgeCGMax;
	double outputEdgeCGAvg;
	double outputEdgeCGStd;
	
	double inputWeightCGMax;
	double inputWeightCGAvg;
	double inputWeightCGStd;
	double outputWeightCGMax;
	double outputWeightCGAvg;
	double outputWeightCGStd;
	
	//Causal Gragh (High Level Variables)
	double inputEdgeHVMax;
	double inputEdgeHVAvg;
	double inputEdgeHVStd;
	double outputEdgeHVMax;
	double outputEdgeHVAvg;
	double outputEdgeHVStd;
	
	double inputWeightHVMax;
	double inputWeightHVAvg;
	double inputWeightHVStd;
	double outputWeightHVMax;
	double outputWeightHVAvg;
	double outputWeightHVStd;
	
	//Domain Transition Graph
	int numberVariablesDTG;
	int totalEdgesDTG;
	int totalWeigthDTG;
	
	double edVaRatioDTG; //totalEdgesDTG/numberVariablesDTG
	double weEdRatioDTG; // totalWeigthDTG/totalEdgesDTG;
	double weVaRatioDTG; // totalWeigthDTG/totalEdgesDTG
	
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
	
	
	void statisticsCG(vector<simplyCG> simplyCG, vector<int> goals, int variables);
	void statisticsDTG(vector<simplyDTG> simplyDTG, vector<int> goals, int variables);
	vector<double> getValuesVector(vector<int> exampleVector, vector<double> stadistics);
	void printVectors(vector<int> inputsInt, vector<int> weightsInt, vector<int> inputsOut, vector<int> weightsOut, vector<int> inputsIntHV, vector<int> weightsIntHV, vector<int> inputsOutHV ,vector<int> weightsOutHV, int variables);
	void transfordValues(vector<Values> valuesDTG);
public:
    Features();
    void generate_cpp_input(ofstream &outfile) const;
    void dump() const;
    void dumpFile(ofstream &outfile) const;
    void dumpSimply() const;
    void dumpFileSimply(ofstream &outfile) const;
    void calculateValueCG(Store store);
    void calculateValueDTG(Store store);
    
	int getNumberVariablesCG();
	int gethighLevelVariablesCG();
	int getTotalEdgesCG();
	int getTotalWeightCG();
	double getveRatio();
	double getweRatio();
	double getwvRatio();
	double gethvRatio();

	double getInputEdgeCGMax();
	double getInputEdgeCGAvg();
	double getInputEdgeCGStd();
	double getOutputEdgeCGMax();
	double getOutputEdgeCGAvg();
	double getOutputEdgeCGStd();

	double getInputWeightCGMax();
	double getInputWeightCGAvg();
	double getInputWeightCGStd();
	double getOutputWeightCGMax();
	double getOutputWeightCGAvg();
	double getOutputWeightCGStd();

	//Causal Gragh (High Level Variables)
	double getInputEdgeHVMax();
	double getInputEdgeHVAvg();
	double getInputEdgeHVStd();
	double getOutputEdgeHVMax();
	double getOutputEdgeHVAvg();
	double getOutputEdgeHVStd();

	double getInputWeightHVMax();
	double getInputWeightHVAvg();
	double getInputWeightHVStd();
	double getOutputWeightHVMax();
	double getOutputWeightHVAvg();
	double getOutputWeightHVStd();

	//Domain Transition Graph
	int getNumberVariablesDTG();
	int getTotalEdgesDTG();
	int getTotalWeigthDTG();

	double getedVaRatioDTG(); //totalEdgesDTG/numberVariablesDTG
	double getweEdRatioDTG(); // totalWeigthDTG/totalEdgesDTG();
	double getweVaRatioDTG(); // totalWeigthDTG/totalEdgesDTG

	double getInputEdgeDTGMax();
	double getInputEdgeDTGAvg();
	double getInputEdgeDTGStd();
	double getOutputEdgeDTGMax();
	double getOutputEdgeDTGAvg();
	double getOutputEdgeDTGStd();

	double getInputWeightDTGMax();
	double getInputWeightDTGAvg();
	double getInputWeightDTGStd();
	double getOutputWeightDTGMax();
	double getOutputWeightDTGAvg();
	double getOutputWeightDTGStd();
	
	void setNumberVariablesCG(int value);
	void sethighLevelVariablesCG(int value);
	void setTotalEdgesCG(int value);
	void setTotalWeightCG(int value);
	void setveRatio(double value);
	void setweRatio(double value);
	void setwvRatio(double value);
	void sethvRatio(double value);
	
	void setInputEdgeCGMax(double value);
	void setInputEdgeCGAvg(double value);
	void setInputEdgeCGStd(double value);
	void setOutputEdgeCGMax(double value);
	void setOutputEdgeCGAvg(double value);
	void setOutputEdgeCGStd(double value);
	
	void setInputWeightCGMax(double value);
	void setInputWeightCGAvg(double value);
	void setInputWeightCGStd(double value);
	void setOutputWeightCGMax(double value);
	void setOutputWeightCGAvg(double value);
	void setOutputWeightCGStd(double value);
	
	//Causal Gragh (High Level Variables)
	void setInputEdgeHVMax(double value);
	void setInputEdgeHVAvg(double value);
	void setInputEdgeHVStd(double value);
	void setOutputEdgeHVMax(double value);
	void setOutputEdgeHVAvg(double value);
	void setOutputEdgeHVStd(double value);
	
	void setInputWeightHVMax(double value);
	void setInputWeightHVAvg(double value);
	void setInputWeightHVStd(double value);
	void setOutputWeightHVMax(double value);
	void setOutputWeightHVAvg(double value);
	void setOutputWeightHVStd(double value);
	
	//Domain Transition Graph
	void setNumberVariablesDTG(int value);
	void setTotalEdgesDTG(int value);
	void setTotalWeigthDTG(int value);
	
	void setedVaRatioDTG(double value); //totalEdgesDTG/numberVariablesDTG
	void setweEdRatioDTG(double value); // totalWeigthDTG/totalEdgesDTG(double value);
	void setweVaRatioDTG(double value); // totalWeigthDTG/totalEdgesDTG
	
	void setInputEdgeDTGMax(double value);
	void setInputEdgeDTGAvg(double value);
	void setInputEdgeDTGStd(double value);
	void setOutputEdgeDTGMax(double value);
	void setOutputEdgeDTGAvg(double value);
	void setOutputEdgeDTGStd(double value);
	
	void setInputWeightDTGMax(double value);
	void setInputWeightDTGAvg(double value);
	void setInputWeightDTGStd(double value);
	void setOutputWeightDTGMax(double value);
	void setOutputWeightDTGAvg(double value);
	void setOutputWeightDTGStd(double value);
};

#endif
