#ifndef DOUBLE_SCALAR_EVALUATOR_H
#define DOUBLE_SCALAR_EVALUATOR_H

#include "evaluator.h"

class DoubleScalarEvaluator : public Evaluator {
public:
    virtual ~DoubleScalarEvaluator() {}

    virtual double get_value() const = 0;
};

#endif
