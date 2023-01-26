#include "double_sum_evaluator.h"

#include <limits>
#include <cassert>

#include "option_parser.h"
#include "plugin.h"

DoubleSumEvaluator::DoubleSumEvaluator(const Options &opts)
    : DoubleCombiningEvaluator(opts.get_list<DoubleScalarEvaluator *>("evals")) {
    cte = opts.get<double>("cte");
}

DoubleSumEvaluator::DoubleSumEvaluator(const std::vector<DoubleScalarEvaluator *> &evals, const double c)
    : DoubleCombiningEvaluator(evals) {
    cte = c;
}

DoubleSumEvaluator::~DoubleSumEvaluator() {
}

int DoubleSumEvaluator::combine_values(const vector<double> &values) {
    double result = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        result += values[i];
//        std::cout << "v[" << i << "]--> " << values[i] << std::endl;
    }
//    std::cout << "cte: " << cte << std::endl;
//    std::cout << "result: " << result << std::endl;
    if (result < 0) {
        result = 0;
    }
    return ((int)result);
}



static ScalarEvaluator *_parse(OptionParser &parser) {
    parser.add_list_option<DoubleScalarEvaluator *>("evals");
    parser.add_option<double>("cte", 0.0);
    Options opts = parser.parse();

    opts.verify_list_non_empty<DoubleScalarEvaluator *>("evals");

    if (parser.dry_run())
        return 0;
    else
        return new DoubleSumEvaluator(opts);
}

static Plugin<ScalarEvaluator> _plugin("double_sum", _parse);
