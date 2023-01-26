#ifndef DOUBLE_SUM_EVALUATOR_H
#define DOUBLE_SUM_EVALUATOR_H

#include "double_combining_evaluator.h"

#include <vector>

class Options;

class DoubleSumEvaluator : public DoubleCombiningEvaluator {
    double cte;
protected:
    virtual int combine_values(const std::vector<double> &values);
public:
    DoubleSumEvaluator(const Options &opts);
    DoubleSumEvaluator(const std::vector<DoubleScalarEvaluator *> &evals, const double c);
    ~DoubleSumEvaluator();
};

#endif
