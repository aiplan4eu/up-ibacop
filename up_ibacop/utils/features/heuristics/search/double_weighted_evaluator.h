#ifndef DOUBLE_WEIGHTED_EVALUATOR_H
#define DOUBLE_WEIGHTED_EVALUATOR_H

#include "scalar_evaluator.h"
#include "double_scalar_evaluator.h"

#include <string>
#include <vector>

class Options;

class DoubleWeightedEvaluator : public DoubleScalarEvaluator {
private:
    ScalarEvaluator *evaluator;
    double w;
    double value;

public:
    DoubleWeightedEvaluator(const Options &opts);
    DoubleWeightedEvaluator(ScalarEvaluator *eval, double weight);
    ~DoubleWeightedEvaluator();

    void evaluate(int g, bool preferred);
    bool is_dead_end() const;
    bool dead_end_is_reliable() const;
    double get_value() const;
    void get_involved_heuristics(std::set<Heuristic *> &hset);
};

#endif
