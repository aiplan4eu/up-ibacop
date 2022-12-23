#include "double_weighted_evaluator.h"

#include <cstdlib>
#include <sstream>

#include "option_parser.h"
#include "plugin.h"

DoubleWeightedEvaluator::DoubleWeightedEvaluator(const Options &opts)
    : evaluator(opts.get_list<ScalarEvaluator *>("evals")[0]),
      w(opts.get<double>("weight")) {
}

DoubleWeightedEvaluator::DoubleWeightedEvaluator(ScalarEvaluator *eval, double weight)
    : evaluator(eval), w(weight) {
}


DoubleWeightedEvaluator::~DoubleWeightedEvaluator() {
}

void DoubleWeightedEvaluator::evaluate(int g, bool preferred) {
    evaluator->evaluate(g, preferred);
    value = w * ((double) evaluator->get_value());
//    std::cout << "w: " << w << ", evaluator: " << evaluator->get_value() << " --> " << value << std::endl;
    // TODO: catch overflow?
}

bool DoubleWeightedEvaluator::is_dead_end() const {
    return evaluator->is_dead_end();
}

bool DoubleWeightedEvaluator::dead_end_is_reliable() const {
    return evaluator->dead_end_is_reliable();
}

double DoubleWeightedEvaluator::get_value() const {
    return value;
}

void DoubleWeightedEvaluator::get_involved_heuristics(std::set<Heuristic *> &hset) {
    evaluator->get_involved_heuristics(hset);
}

static DoubleScalarEvaluator *_parse(OptionParser &parser) {
    parser.add_list_option<ScalarEvaluator *>("evals");
    parser.add_option<double>("weight");
    Options opts = parser.parse();
    if (parser.dry_run())
        return 0;
    else
        return new DoubleWeightedEvaluator(opts);
}

static Plugin<DoubleScalarEvaluator> _plugin("double_weight", _parse);
