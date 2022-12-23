#ifndef DOUBLE_COMBINING_EVALUATOR_H
#define DOUBLE_COMBINING_EVALUATOR_H

#include "scalar_evaluator.h"
#include "double_scalar_evaluator.h"

#include <set>
#include <string>
#include <vector>

/*
  CombiningEvaluator is the base class for SumEvaluator and
  MaxEvaluator, which captures the common aspects of their behaviour.
  */

class DoubleCombiningEvaluator : public ScalarEvaluator {
    std::vector<DoubleScalarEvaluator *> subevaluators;
    std::vector<double> subevaluator_values;
    int value;
    bool dead_end;
    bool dead_end_reliable;
protected:
    virtual int combine_values(const std::vector<double> &values) = 0;
public:
    DoubleCombiningEvaluator(const std::vector<DoubleScalarEvaluator *> &subevaluators_);
    ~DoubleCombiningEvaluator();

    virtual void evaluate(int g, bool preferred);
    virtual bool is_dead_end() const;
    virtual bool dead_end_is_reliable() const;
    virtual int get_value() const;
    virtual void get_involved_heuristics(std::set<Heuristic *> &hset);
};

#endif
