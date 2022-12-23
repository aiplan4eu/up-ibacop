#include "training_tasks.h"

#include "option_parser.h"

#include <fstream>
#include <string>

void TrainingTasks::solution_path_data_collection(const State &g_initial_state, const int cost_type) {

	std::cout << "-------------------------------------------------- TrainingTasks begin" << std::endl;

    SearchSpace search_space(((cost_type==0)?NORMAL:((cost_type==1)?ONE:PLUSONE)));

        vector<Heuristic *> heuristics(9);

        Options default_opt;
        default_opt.set<int>("cost_type", cost_type);

        Options lm_opt;
        lm_opt.set("cost_type", cost_type);
        lm_opt.set("lm_cost_type", cost_type);
        lm_opt.set("reasonable_orders", true);
        lm_opt.set("only_causal_landmarks", false);
        lm_opt.set("disjunctive_landmarks", true);
        lm_opt.set("conjunctive_landmarks", false);
        lm_opt.set("no_orders", false);

        Exploration* explor = new Exploration(lm_opt);

        lm_opt.set<Exploration *>("explor", explor);
        LandmarkFactoryRpgSasp lm_graph_factory(lm_opt);
        LandmarkGraph *graph = lm_graph_factory.compute_lm_graph();

        Options lmcount_opt;
        lmcount_opt.set("cost_type", cost_type);
        lmcount_opt.set("admissible", false);
        lmcount_opt.set("optimal", false);
        lmcount_opt.set("pref", false);
        lmcount_opt.set("alm", true);
        lmcount_opt.set("lm_graph", graph);

        heuristics[0] = new AdditiveHeuristic(default_opt);
        heuristics[1] = new BlindSearchHeuristic(default_opt);
        heuristics[2] = new CGHeuristic(default_opt);
        heuristics[3] = new cea_heuristic::ContextEnhancedAdditiveHeuristic(default_opt);
        heuristics[4] = new FFHeuristic(default_opt);
        heuristics[5] = new GoalCountHeuristic(default_opt);
        heuristics[6] = new LandmarkCountHeuristic(lmcount_opt);
        heuristics[7] = new LandmarkCutHeuristic(default_opt);
        heuristics[8] = new HSPMaxHeuristic(default_opt);

        ofstream results;
//        results.open("tmp_results", ofstream::app);
        results.open("tmp_results", ofstream::trunc);
//results << "--- path_data_collection ---" << std::endl;
        SearchNode node = search_space.get_node(g_initial_state);
        State s = node.get_state();
        
//////////////////////////////////

    for (int h = 0; h < heuristics.size(); h++) {
        heuristics[h]->evaluate(s);
        results << heuristics[h]->get_heuristic();
        if (h < heuristics.size() - 1) {
	    results << ",";
	}
    }
    results << std::endl;

//////////////////////////////////
        
        results.close();

        for (int h = 0; h < heuristics.size(); h++) {
            delete heuristics[h];
        }
        delete explor;

	std::cout << "-------------------------------------------------- TrainingTasks end" << std::endl;
}
