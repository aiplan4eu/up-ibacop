/**
** @author Isabel Cenamor <icenamor@inf.uc3m.es> 2013
**/
#include "Features.h"
#include "Store.h"

#include "helper_functions.h"

#include <cassert>
#include <math.h>    
using namespace std;

Features::Features() {

	numberVariablesCG = -1;
	highLevelVariablesCG = -1;
	totalEdgesCG = -1;
	totalWeightCG = -1;
	
	veRatio = -1;
	weRatio = -1;
	wvRatio = -1;
	hvRatio = -1;
	
	inputEdgeCGMax = -1;
	inputEdgeCGAvg = -1;
	inputEdgeCGStd = -1;
	outputEdgeCGMax = -1;
	outputEdgeCGAvg = -1;
	outputEdgeCGStd = -1;
	
	inputWeightCGMax = -1;
	inputWeightCGAvg = -1;
	inputWeightCGStd = -1;
	outputWeightCGMax = -1;
	outputWeightCGAvg = -1;
	outputWeightCGStd = -1;
	
	//Causal Gragh (High Level Variables)
	inputEdgeHVMax = -1;
	inputEdgeHVAvg = -1;
	inputEdgeHVStd = -1;
	outputEdgeHVMax = -1;
	outputEdgeHVAvg = -1;
	outputEdgeHVStd = -1;
	
	inputWeightHVMax = -1;
	inputWeightHVAvg = -1;
	inputWeightHVStd = -1;
	outputWeightHVMax = -1;
	outputWeightHVAvg = -1;
	outputWeightHVStd = -1;
	
	//Domain Transition Graph
	numberVariablesDTG = -1;
	totalEdgesDTG = -1;
	totalWeigthDTG = -1;
	
	edVaRatioDTG = (totalEdgesDTG/numberVariablesDTG); //totalEdgesDTG/numberVariablesDTG
	weEdRatioDTG = (totalWeigthDTG/totalEdgesDTG ); // totalWeigthDTG/totalEdgesDTG = -1;
	weVaRatioDTG = (totalWeigthDTG/totalEdgesDTG); // totalWeigthDTG/totalEdgesDTG
	
	inputEdgeDTGMax = -1;
	inputEdgeDTGAvg = -1;
	inputEdgeDTGStd = -1;
	outputEdgeDTGMax = -1;
	outputEdgeDTGAvg = -1;
	outputEdgeDTGStd = -1;
	
	inputWeightDTGMax = -1;
	inputWeightDTGAvg = -1;
	inputWeightDTGStd = -1;
	outputWeightDTGMax = -1;
	outputWeightDTGAvg = -1;
	outputWeightDTGStd = -1;
	
}

void Features::dump() const {
	cout<<numberVariablesCG<<","<<highLevelVariablesCG<<", "<<totalEdgesCG<<", "<<totalWeightCG<<", "<<veRatio<<", "<<weRatio<<", "<<wvRatio<<", "<<hvRatio<<", "<<endl;
	cout<<inputEdgeCGMax<<", "<<inputEdgeCGAvg<<", "<<inputEdgeCGStd<<", "<<outputEdgeCGMax<<", "<<outputEdgeCGAvg<<", "<<outputEdgeCGStd<<", "<<inputWeightCGMax<<", "<<inputWeightCGAvg<<", "<<inputWeightCGStd<<","<<outputWeightCGMax<<", "<<outputWeightCGAvg<<", "<<outputWeightCGStd<<endl;
	cout<<", "<<inputEdgeHVMax<<", "<<inputEdgeHVAvg<<", "<<inputEdgeHVStd<<", "<<outputEdgeHVMax<<", "<<outputEdgeHVAvg<<", "<<outputEdgeHVStd<<", "<<inputWeightHVMax<<", "<<inputWeightHVAvg<<", "<<inputWeightHVStd<<", "<<outputWeightHVMax<<", "<<outputWeightHVAvg<<","<<outputWeightHVStd<<","<<endl;
	cout<<numberVariablesDTG<<", "<<totalEdgesDTG<<", "<<totalWeigthDTG<<", "<<edVaRatioDTG<<", "<<weEdRatioDTG<<", "<<weVaRatioDTG<<", "<<inputEdgeDTGMax<<", "<<inputEdgeDTGAvg<<", "<<inputEdgeDTGStd<<", "<<endl;
	cout<<outputEdgeDTGMax<<", "<<outputEdgeDTGAvg<<", "<<outputEdgeDTGStd<<", "<<inputWeightDTGMax<<", "<<inputWeightDTGAvg<<","<<inputWeightDTGStd<<", "<<outputWeightDTGMax<<","<<outputWeightDTGAvg<<", "<<outputWeightDTGStd<<endl;
cout<<"--"<<endl;
}

void Features::dumpSimply() const {
	cout<<numberVariablesCG<<","<<hvRatio<<", ";
	
	cout<<inputEdgeCGStd<<", "<<outputEdgeCGAvg<<", "<<inputWeightCGMax<<", "<<inputWeightCGAvg<<",";
	
	cout<<outputEdgeHVStd<<", "<<inputWeightHVMax<<",";
	
	cout<<numberVariablesDTG<<", "<<totalEdgesDTG<<", ";
	
	cout<<inputWeightDTGMax<<endl;
	
}


void Features::dumpFile(ofstream &outfile) const {

    	outfile.open("features.arff", ios::out);
	outfile<<numberVariablesCG<<","<<highLevelVariablesCG<<", "<<totalEdgesCG<<", "<<totalWeightCG<<", "<<veRatio<<", "<<weRatio<<", "<<wvRatio<<", "<<hvRatio<<", "<<inputEdgeCGMax<<", "<<inputEdgeCGAvg<<", "<<inputEdgeCGStd<<", "<<outputEdgeCGMax<<", "<<outputEdgeCGAvg<<", "<<outputEdgeCGStd<<", "<<inputWeightCGMax<<", "<<inputWeightCGAvg<<", "<<inputWeightCGStd<<","<<outputWeightCGMax<<", "<<outputWeightCGAvg<<", "<<outputWeightCGStd<<", "<<inputEdgeHVMax<<", "<<inputEdgeHVAvg<<", "<<inputEdgeHVStd<<", "<<outputEdgeHVMax<<", "<<outputEdgeHVAvg<<", "<<outputEdgeHVStd<<", "<<inputWeightHVMax<<", "<<inputWeightHVAvg<<", "<<inputWeightHVStd<<", "<<outputWeightHVMax<<", "<<outputWeightHVAvg<<","<<outputWeightHVStd<<","<<numberVariablesDTG<<", "<<totalEdgesDTG<<", "<<totalWeigthDTG<<", "<<edVaRatioDTG<<", "<<weEdRatioDTG<<", "<<weVaRatioDTG<<", "<<inputEdgeDTGMax<<", "<<inputEdgeDTGAvg<<", "<<inputEdgeDTGStd<<", "<<outputEdgeDTGMax<<", "<<outputEdgeDTGAvg<<", "<<outputEdgeDTGStd<<", "<<inputWeightDTGMax<<", "<<inputWeightDTGAvg<<","<<inputWeightDTGStd<<", "<<outputWeightDTGMax<<","<<outputWeightDTGAvg<<", "<<outputWeightDTGStd<<endl;
	
	outfile.close();
}

void Features::dumpFileSimply(ofstream &outfile) const {
    outfile.open("featuresSimply.arff", ios::out);
	outfile<<numberVariablesCG<<","<<hvRatio<<", ";
	outfile<<inputEdgeCGStd<<", "<<outputEdgeCGAvg<<", "<<inputWeightCGMax<<", "<<inputWeightCGAvg<<",";
	outfile<<outputEdgeHVStd<<", "<<inputWeightHVMax<<",";
	outfile<<numberVariablesDTG<<", "<<totalEdgesDTG<<", ";
	outfile<<inputWeightDTGMax<<endl;
	outfile.close();
}

void Features::generate_cpp_input(ofstream &outfile) const {
  outfile << "begin_variable" << endl;
  outfile << "end_variable" << endl;
}

/**
**return the maximun, the average and the standard desviation for each vector
**/
vector<double> Features::getValuesVector(vector<int> exampleVector, vector<double> stadistics){
	double max = 0.0;
	double total = 0.0;
	double average = 0.0;
	double std = 0.0;
	stadistics.clear();
	for (int i = 0; i < exampleVector.size(); i++){
		total += exampleVector[i];
		if( exampleVector[i] > max){
			max = exampleVector[i];
		}
	}
	average = double(total/exampleVector.size());
	double auxiliar = 0.0;
	for (int i = 0; i < exampleVector.size(); i++){
		double rest = double(exampleVector[i] - average);
		double square = pow(rest, 2);
		auxiliar += square;
	}
	auxiliar = double(double(auxiliar)/double(exampleVector.size()));
	std = sqrt(auxiliar);
	stadistics.push_back(max);
	stadistics.push_back(average);
	stadistics.push_back(std);
	return stadistics;
	 
}

//the statistics of CG
void Features::statisticsCG(vector<simplyCG> simplyCG, vector<int> goals, int variables){

	vector<int> inputsInt;
	vector<int> weightsInt;
	vector<int> inputsOut;
	vector<int> weightsOut;
	vector<int> inputsIntHV;
	vector<int> weightsIntHV;
	vector<int> inputsOutHV;
	vector<int> weightsOutHV;
	
	for(int i = 0; i < variables; i++){
		inputsInt.push_back(0);
		weightsInt.push_back(0);
		inputsOut.push_back(0);
		weightsOut.push_back(0);
		inputsIntHV.push_back(0);
		weightsIntHV.push_back(0);
		inputsOutHV.push_back(0);
		weightsOutHV.push_back(0);
	}
	for (int v = 0; v < variables; v++){
		for (int i = 0; i < simplyCG.size(); i++){
			if(simplyCG[i].getSource() == v){
				inputsInt[v] = simplyCG[i].getSucessor();
				weightsInt[v] = weightsInt[v] + simplyCG[i].getValue();
			}
			if(simplyCG[i].getTarget() == v){
				inputsOut[v] = inputsOut[v] + 1; //pensar
				weightsOut[v] = weightsOut[v] + simplyCG[i].getValue();
			}
			for (int h = 0; h < goals.size(); h++){
				if(goals[h] == simplyCG[i].getSource() &&  simplyCG[i].getSource() == v && goals[h] == v){
					inputsIntHV[v] = simplyCG[i].getSucessor();
					weightsIntHV[v] = weightsIntHV[v] + simplyCG[i].getValue();
				}
				if(goals[h] == simplyCG[i].getTarget() &&  simplyCG[i].getTarget() == v && goals[h] == v){
					inputsOutHV[v] = inputsOutHV[v] + simplyCG[i].getValue();
					weightsOutHV[v] = weightsOutHV[v] + simplyCG[i].getValue();
				}
			}
		}
	}
	vector<double> values;
	values = getValuesVector(inputsInt, values);
	setInputEdgeCGMax(values[0]);
	setInputEdgeCGAvg(values[1]);
	setInputEdgeCGStd(values[2]);
	values.clear();
	values = getValuesVector(weightsInt,values);
	setInputWeightCGMax(values[0]);
	setInputWeightCGAvg(values[1]);
	setInputWeightCGStd(values[2]);
	values.clear();
	values = getValuesVector(inputsOut,values);
	setOutputEdgeCGMax(values[0]);
	setOutputEdgeCGAvg(values[1]);
	setOutputEdgeCGStd(values[2]);
	values.clear();
	values = getValuesVector(weightsOut,values);
	setOutputWeightCGMax(values[0]);
	setOutputWeightCGAvg(values[1]);
	setOutputWeightCGStd(values[2]);
	
	values.clear();
	values = getValuesVector(inputsIntHV,values);
	setInputEdgeHVMax(values[0]);
	setInputEdgeHVAvg(values[1]);
	setInputEdgeHVStd(values[2]);
	values.clear();
	values = getValuesVector(weightsIntHV,values);
	setInputWeightHVMax(values[0]);
	setInputWeightHVAvg(values[1]);
	setInputWeightHVStd(values[2]);
	values.clear();
	values = getValuesVector(inputsOutHV,values);
	setOutputEdgeHVMax(values[0]);
	setOutputEdgeHVAvg(values[1]);
	setOutputEdgeHVStd(values[2]);
	values.clear();
	values = getValuesVector(weightsOutHV,values);
	setOutputWeightHVMax(values[0]);
	setOutputWeightHVAvg(values[1]);
	setOutputWeightHVStd(values[2]);
	values.clear();
	
}



/**
** 
**/
void Features::transfordValues(vector<Values> valuesDTG){
	double totalWeigthIn = 0;
	double totalWeigthOut = 0;
	double totalIn = 0;
	double totalOut = 0;
	
	double std_totalWeigthIn = 0;
	double std_totalWeigthOut = 0;
	double std_totalIn = 0;
	double std_totalOut = 0;
	
	double max_totalWeigthIn = 0;
	double max_totalWeigthOut = 0;
	double max_totalIn = 0;
	double max_totalOut = 0;
	
	for(int i = 0; i < valuesDTG.size();i++){
	
		totalWeigthIn += valuesDTG[i].inputWeightDTGAvg;
		totalWeigthOut += valuesDTG[i].outputWeightDTGAvg;
		totalIn += valuesDTG[i].inputEdgeDTGAvg;
		totalOut += valuesDTG[i].outputEdgeDTGAvg;
		std_totalWeigthIn = valuesDTG[i].inputWeightDTGStd;
		std_totalWeigthOut = valuesDTG[i].outputWeightDTGStd;
		std_totalIn = valuesDTG[i].inputEdgeDTGStd;
		std_totalOut = valuesDTG[i].outputEdgeDTGStd;
		
		if(max_totalIn < valuesDTG[i].inputEdgeDTGMax ){
			max_totalIn = valuesDTG[i].inputEdgeDTGMax;
		}
		if(max_totalOut < valuesDTG[i].outputEdgeDTGMax ){
			max_totalOut = valuesDTG[i].outputEdgeDTGMax ;
		}
		if(max_totalWeigthIn < valuesDTG[i].inputWeightDTGMax ){
			max_totalWeigthIn = valuesDTG[i].inputWeightDTGMax;
		}
		if(max_totalWeigthOut < valuesDTG[i].outputWeightDTGMax ){
			max_totalWeigthOut = valuesDTG[i].outputWeightDTGMax;
		}
	}
	
	double averageIn = double(double(totalIn)/double(valuesDTG.size()));
	double averageOut = double(double(totalOut)/double(valuesDTG.size()));
	double averageWeIn = double(double(totalWeigthIn)/double(valuesDTG.size()));
	double avergaeWeOut = double(double(totalWeigthOut)/double(valuesDTG.size()));
	
	double stdIn = double(double(std_totalIn)/double(valuesDTG.size()));
	double stdOut = double(double(std_totalOut)/double(valuesDTG.size()));
	double stdWeIn = double(double(std_totalWeigthIn)/double(valuesDTG.size()));
	double stdWeOut = double(double(std_totalWeigthOut)/double(valuesDTG.size()));
	
	double auxiliarinput = 0.0;
	double auxiliaroutput = 0.0;
	double  auxiliarwein = 0.0;
	double auxiliarweout = 0.0;
	
	for(int i = 0; i < valuesDTG.size();i++){
		auxiliarwein = pow(double(valuesDTG[i].inputWeightDTGStd - stdWeIn), 2);
		auxiliarweout = pow(double(valuesDTG[i].outputWeightDTGStd - stdWeOut), 2);
		auxiliarinput = pow(double(valuesDTG[i].inputEdgeDTGStd - stdIn), 2);
		auxiliaroutput = pow(double(valuesDTG[i].outputEdgeDTGStd - stdOut), 2);
	}
	
	auxiliarwein = sqrt(auxiliarwein/double(valuesDTG.size()));
	auxiliarweout = sqrt(auxiliarweout/double(valuesDTG.size()));
	auxiliarinput = sqrt(auxiliarinput/double(valuesDTG.size()));
	auxiliaroutput = sqrt(auxiliaroutput/double(valuesDTG.size()));
		
	setInputEdgeDTGMax(max_totalIn);
	setInputEdgeDTGAvg(averageIn);
	setInputEdgeDTGStd(auxiliarinput);
	setOutputEdgeDTGMax(max_totalOut);
	setOutputEdgeDTGAvg(averageOut);
	setOutputEdgeDTGStd(auxiliaroutput);
	setInputWeightDTGMax(max_totalWeigthIn);
	setInputWeightDTGAvg(averageWeIn);
	setInputWeightDTGStd(auxiliarwein);
	setOutputWeightDTGMax(max_totalWeigthOut);
	setOutputWeightDTGAvg(avergaeWeOut);
	setOutputWeightDTGStd(auxiliaroutput);
}




void Features::statisticsDTG(vector<simplyDTG> simplydtg, vector<int> goals, int variables){
	cout<<"statistics DTG"<<endl;
	double totalVariablesDTG_general =  0.0;
	double totalWeigthDTG_general = 0.0;
	vector<Values> valuesDTG;
	
	double numberVariablesDTGAvg = 0;
	
	//double totalEdgesDTGAvg = 0;
		
	//double totalWeigthDTGAvg = 0;
	
	for(int v = 0; v < variables; v++){
		vector<simplyDTG> pValores;
		double totalWeigthDTG = 0.0;
		double numberVariablesDTG = 0.0;
		for (int i = 0; i < simplydtg.size(); i++){
			if(v == simplydtg[i].getVariable()){
				totalWeigthDTG += simplydtg[i].getConditions();
				if(simplydtg[i].getTarget() > simplydtg[i].getSource()){
					numberVariablesDTG = simplydtg[i].getTarget() + 1;
				}else{
					numberVariablesDTG = simplydtg[i].getSource() + 1;
				}
				pValores.push_back(simplydtg[i]);
			}
		}//end simplyDTG
		vector<int> inputsInt;
		vector<int> weightsInt;
		vector<int> inputsOut;
		vector<int> weightsOut;
		vector<int> inputsIntHV;
		vector<int> weightsIntHV;
		vector<int> inputsOutHV;
		vector<int> weightsOutHV;

		for(int i = 0; i < numberVariablesDTG; i++){
			inputsInt.push_back(0);
			weightsInt.push_back(0);
			inputsOut.push_back(0);
			weightsOut.push_back(0);
			inputsIntHV.push_back(0);
			weightsIntHV.push_back(0);
			inputsOutHV.push_back(0);
			weightsOutHV.push_back(0);
		}
		
		numberVariablesDTGAvg += numberVariablesDTG;
		
		for (int j = 0; j <numberVariablesDTG; j++){
			for(int i = 0; i < pValores.size(); i++){
				if(simplydtg[i].getSource() == j){
					inputsInt[j] = inputsInt[j] + 1;
					weightsInt[j] = weightsInt[j] + simplydtg[i].getConditions();
				}
				if(simplydtg[i].getTarget() == j){
					inputsOut[j] = inputsOut[j] + 1; //pensar
					weightsOut[j] = weightsOut[j] + simplydtg[i].getConditions();
				}
				for (int h = 0; h < goals.size(); h++){
					if(goals[h] == simplydtg[i].getSource() &&  simplydtg[i].getSource() == v && goals[h] == j){
						inputsIntHV[j] = inputsIntHV[j] + 1;
						weightsIntHV[j] = weightsIntHV[j] + simplydtg[i].getConditions();
					}
					if(goals[h] == simplydtg[i].getTarget() &&  simplydtg[i].getTarget() == v && goals[h] == j){
						inputsOutHV[j] = inputsOutHV[j] + simplydtg[i].getConditions();
						weightsOutHV[j] = weightsOutHV[j] + simplydtg[i].getConditions();
					}
				}
			}
		}
	vector<double> values;
	Values val = Values();
	val.var = v;
	values = getValuesVector(inputsInt, values);
	val.inputEdgeDTGMax = values[0];
	val.inputEdgeDTGAvg = values[1];
	val.inputEdgeDTGStd = values[2];
	values.clear();
	values = getValuesVector(weightsInt,values);
	val.inputWeightDTGMax = values[0];
	val.inputWeightDTGAvg = values[1];
	val.inputWeightDTGStd = values[2];
	values.clear();
	values = getValuesVector(inputsOut,values);
	val.outputEdgeDTGMax = values[0];
	val.outputEdgeDTGAvg = values[1];
	val.outputEdgeDTGStd = values[2];
	values.clear();
	values = getValuesVector(weightsOut,values);
	val.outputWeightDTGMax = values[0];
	val.outputWeightDTGAvg = values[1];
	val.outputWeightDTGStd = values[2];
	values.clear();
	valuesDTG.push_back(val);
	//val.print();
	totalWeigthDTG_general += totalWeigthDTG;
	totalVariablesDTG_general += numberVariablesDTG;
	
	
	}//end var
	setNumberVariablesDTG(totalVariablesDTG_general);
	setTotalEdgesDTG(simplydtg.size());
	setTotalWeigthDTG(totalWeigthDTG_general); ///totalWeigthDTG

	setedVaRatioDTG(double(double(simplydtg.size())/double(totalWeigthDTG_general)));
	setweEdRatioDTG(double(double(totalVariablesDTG_general)/double(simplydtg.size())));
	setweVaRatioDTG(double(double(totalVariablesDTG_general)/double(totalVariablesDTG_general)));
	
	/*for (int i = 0; i < valuesDTG.size(); i++){
		valuesDTG[i].print();
	}*/
	transfordValues(valuesDTG);
}

void Features::calculateValueCG(Store store){
	// general CG
	setNumberVariablesCG(store.getVariablesCG());
	sethighLevelVariablesCG(store.getGoals().size());
	setTotalEdgesCG(store.getCausalGraph().size());
	setTotalWeightCG(store.getTotalWeightCG());

	double ve = double( double(getNumberVariablesCG())/double(getTotalEdgesCG()));
	double we = double( double(getTotalWeightCG())/ double(getTotalEdgesCG()));
	double wv = double( double(getTotalWeightCG())/ double(getNumberVariablesCG()));
	double hv = double( double(gethighLevelVariablesCG())/ double(getNumberVariablesCG()));
	setveRatio(ve);
	setweRatio(we);
	setwvRatio(wv);
	sethvRatio(hv);
	// CG
	statisticsCG(store.getCausalGraph(), store.getGoals(), getNumberVariablesCG());
	//dump();
}
void Features::calculateValueDTG(Store store){
	int variables = getNumberVariablesCG();
	totalEdgesDTG = store.getDomainTransitionGraph().size();
	statisticsDTG(store.getDomainTransitionGraph(), store.getGoals(), variables);
}

	int Features::getNumberVariablesCG(){
		return numberVariablesCG;
	}
	int Features::gethighLevelVariablesCG(){
		return highLevelVariablesCG;
	}
	int Features::getTotalEdgesCG(){
		return totalEdgesCG;
	}
	int Features::getTotalWeightCG(){
		return totalWeightCG;
	}
	double Features::getveRatio(){
		return veRatio;
	}
	double Features::getweRatio(){
		return weRatio;
	}
	double Features::getwvRatio(){
		return wvRatio;
	}
	double Features::gethvRatio(){
		return hvRatio;
	}

	double Features::getInputEdgeCGMax(){
		return inputEdgeCGMax;
	}
	double Features::getInputEdgeCGAvg(){
		return inputEdgeCGAvg;
	}
	double Features::getInputEdgeCGStd(){
		return inputEdgeCGStd;
	}
	double Features::getOutputEdgeCGMax(){
		return outputEdgeCGMax;
	}
	double Features::getOutputEdgeCGAvg(){
		return outputEdgeCGAvg;
	}
	double Features::getOutputEdgeCGStd(){
		return outputEdgeCGStd;
	}

	double Features::getInputWeightCGMax(){
		return inputWeightCGMax;
	}
	double Features::getInputWeightCGAvg(){
		return inputWeightCGAvg;
	}
	double Features::getInputWeightCGStd(){
		return inputWeightCGStd;
	}
	double Features::getOutputWeightCGMax(){
		return outputWeightCGMax;
	}
	double Features::getOutputWeightCGAvg(){
		return outputWeightCGAvg;
	}
	double Features::getOutputWeightCGStd(){
		return outputWeightCGStd;
	}

	//Causal Gragh (High Level Variables)
	double Features::getInputEdgeHVMax(){
		return inputEdgeHVMax;
	}
	double Features::getInputEdgeHVAvg(){
		return inputEdgeHVAvg;
	}
	double Features::getInputEdgeHVStd(){
		return inputEdgeHVStd;
	}
	double Features::getOutputEdgeHVMax(){
		return outputEdgeHVMax;
	}
	double Features::getOutputEdgeHVAvg(){
		return outputEdgeHVAvg;
	}
	double Features::getOutputEdgeHVStd(){
		return outputEdgeHVStd;
	}

	double Features::getInputWeightHVMax(){
		return inputWeightHVMax;
	}
	double Features::getInputWeightHVAvg(){
		return inputWeightHVAvg;
	}
	double Features::getInputWeightHVStd(){
		return inputWeightHVStd;
	}
	double Features::getOutputWeightHVMax(){
		return outputWeightHVMax;
	}
	double Features::getOutputWeightHVAvg(){
		return outputWeightHVAvg;
	}
	double Features::getOutputWeightHVStd(){
		return outputWeightHVStd;
	}

	//Domain Transition Graph
	int Features::getNumberVariablesDTG(){
		return numberVariablesDTG;
	}
	int Features::getTotalEdgesDTG(){
		return totalEdgesDTG;
	}
	int Features::getTotalWeigthDTG(){
		return totalWeigthDTG;
	}

	double Features::getedVaRatioDTG(){
		return edVaRatioDTG;
	} 
	double Features::getweEdRatioDTG(){
		return weEdRatioDTG;
	}
	double Features::getweVaRatioDTG(){
		return weVaRatioDTG;
	} 

	double Features::getInputEdgeDTGMax(){
		return inputEdgeDTGMax;
	}
	double Features::getInputEdgeDTGAvg(){
		return inputEdgeDTGAvg;
	}
	double Features::getInputEdgeDTGStd(){
		return inputEdgeDTGStd;
	}
	double Features::getOutputEdgeDTGMax(){
		return outputEdgeDTGMax;
	}
	double Features::getOutputEdgeDTGAvg(){
		return outputEdgeDTGAvg;
	}
	double Features::getOutputEdgeDTGStd(){
		return outputEdgeDTGStd;
	}

	double Features::getInputWeightDTGMax(){
		return inputWeightDTGMax;
	}
	double Features::getInputWeightDTGAvg(){
		return inputWeightDTGAvg;
	}
	double Features::getInputWeightDTGStd(){
		return inputWeightDTGStd;
	}
	double Features::getOutputWeightDTGMax(){
		return outputWeightDTGMax;
	}
	double Features::getOutputWeightDTGAvg(){
		return outputWeightDTGAvg;
	}
	double Features::getOutputWeightDTGStd(){
		return outputWeightDTGStd;	
	}
	void Features::setNumberVariablesCG(int value){
		numberVariablesCG =  value;
	}
	void Features::sethighLevelVariablesCG(int value){
		highLevelVariablesCG =  value;
	}
	void Features::setTotalEdgesCG(int value){
		totalEdgesCG =  value;
	}
	void Features::setTotalWeightCG(int value){
		totalWeightCG =  value;
	}
	void Features::setveRatio(double value){
		veRatio =  value;
	}
	void Features::setweRatio(double value){
		weRatio =  value;
	}
	void Features::setwvRatio(double value){
		wvRatio =  value;
	}
	void Features::sethvRatio(double value){
		hvRatio =  value;
	}

	void Features::setInputEdgeCGMax(double value){
		inputEdgeCGMax =  value;
	}
	void Features::setInputEdgeCGAvg(double value){
		inputEdgeCGAvg =  value;
	}
	void Features::setInputEdgeCGStd(double value){
		inputEdgeCGStd =  value;
	}
	void Features::setOutputEdgeCGMax(double value){
		outputEdgeCGMax =  value;
	}
	void Features::setOutputEdgeCGAvg(double value){
		outputEdgeCGAvg =  value;
	}
	void Features::setOutputEdgeCGStd(double value){
		outputEdgeCGStd =  value;
	}

	void Features::setInputWeightCGMax(double value){
		inputWeightCGMax =  value;
	}
	void Features::setInputWeightCGAvg(double value){
		inputWeightCGAvg =  value;
	}
	void Features::setInputWeightCGStd(double value){
		inputWeightCGStd =  value;
	}
	void Features::setOutputWeightCGMax(double value){
		outputWeightCGMax =  value;
	}
	void Features::setOutputWeightCGAvg(double value){
		outputWeightCGAvg =  value;
	}
	void Features::setOutputWeightCGStd(double value){
		outputWeightCGStd =  value;
	}

	//Causal Gragh (High Level Variables)
	void Features::setInputEdgeHVMax(double value){
		inputEdgeHVMax =  value;
	}
	void Features::setInputEdgeHVAvg(double value){
		inputEdgeHVAvg =  value;
	}
	void Features::setInputEdgeHVStd(double value){
		inputEdgeHVStd =  value;
	}
	void Features::setOutputEdgeHVMax(double value){
		outputEdgeHVMax =  value;
	}
	void Features::setOutputEdgeHVAvg(double value){
		outputEdgeHVAvg =  value;
	}
	void Features::setOutputEdgeHVStd(double value){
		outputEdgeHVStd =  value;
	}

	void Features::setInputWeightHVMax(double value){
		inputWeightHVMax =  value;
	}
	void Features::setInputWeightHVAvg(double value){
		inputWeightHVAvg =  value;
	}
	void Features::setInputWeightHVStd(double value){
		inputWeightHVStd =  value;
	}
	void Features::setOutputWeightHVMax(double value){
		outputWeightHVMax =  value;
	}
	void Features::setOutputWeightHVAvg(double value){
		outputWeightHVAvg =  value;
	}
	void Features::setOutputWeightHVStd(double value){
		outputWeightHVStd =  value;
	}
	//Domain Transition Graph
	void Features::setNumberVariablesDTG(int value){
		numberVariablesDTG =  value;
	}
	void Features::setTotalEdgesDTG(int value){
		totalEdgesDTG =  value;
	}
	void Features::setTotalWeigthDTG(int value){
		totalWeigthDTG =  value;
	}

	void Features::setedVaRatioDTG(double value){
		edVaRatioDTG =  value;
	} 
	void Features::setweEdRatioDTG(double value){
		weEdRatioDTG =  value;
	}
	void Features::setweVaRatioDTG(double value){
		weVaRatioDTG =  value;
	} 

	void Features::setInputEdgeDTGMax(double value){
		inputEdgeDTGMax =  value;
	}
	void Features::setInputEdgeDTGAvg(double value){
		inputEdgeDTGAvg =  value;
	}
	void Features::setInputEdgeDTGStd(double value){
		inputEdgeDTGStd =  value;
	}
	void Features::setOutputEdgeDTGMax(double value){
		outputEdgeDTGMax =  value;
	}
	void Features::setOutputEdgeDTGAvg(double value){
		outputEdgeDTGAvg =  value;
	}
	void Features::setOutputEdgeDTGStd(double value){
		outputEdgeDTGStd =  value;
	}

	void Features::setInputWeightDTGMax(double value){
		inputWeightDTGMax =  value;
	}
	void Features::setInputWeightDTGAvg(double value){
		inputWeightDTGAvg =  value;
	}
	void Features::setInputWeightDTGStd(double value){
		inputWeightDTGStd =  value;
	}
	void Features::setOutputWeightDTGMax(double value){
		outputWeightDTGMax =  value;
	}
	void Features::setOutputWeightDTGAvg(double value){
		outputWeightDTGAvg =  value;
	}
	void Features::setOutputWeightDTGStd(double value){

		outputWeightDTGStd =  value;	
	}


//Auxiliar for see the values.
void Features::printVectors(vector<int> inputsInt, vector<int> weightsInt, vector<int> inputsOut, vector<int> weightsOut, vector<int> inputsIntHV, vector<int> weightsIntHV, vector<int> inputsOutHV ,vector<int> weightsOutHV, int variables){
	for(int i = 0; i < variables; i++){
	cout<<" "<<i;
	}
	cout<<endl;
	for(int i = 0; i < variables; i++){
	cout<<" "<<inputsInt[i];
	}
	cout<<endl;
		for(int i = 0; i < variables; i++){
	cout<<" "<<weightsInt[i];
	}
	cout<<endl;
		for(int i = 0; i < variables; i++){
	cout<<" "<<inputsOut[i];
	}
	cout<<endl;
		for(int i = 0; i < variables; i++){
	cout<<" "<<weightsOut[i];
	}
	cout<<endl;
		for(int i = 0; i < variables; i++){
	cout<<" "<<inputsIntHV[i];
	}
	cout<<endl;
		for(int i = 0; i < variables; i++){
	cout<<" "<<weightsIntHV[i];
	}
	cout<<endl;
		for(int i = 0; i < variables; i++){
	cout<<" "<<inputsOutHV[i];
	}
	cout<<endl;
		for(int i = 0; i < variables; i++){
	cout<<" "<<weightsOutHV[i];
	}
	cout<<endl;
}

