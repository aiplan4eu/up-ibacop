#ifndef TRAINING_TASKS_H
#define TRAINING_TASKS_H

#include <vector>

#include "heuristic.h"
#include "additive_heuristic.h"
#include "blind_search_heuristic.h"
#include "cg_heuristic.h"
#include "cea_heuristic.h"
#include "ff_heuristic.h"
#include "goal_count_heuristic.h"
#include "landmarks/landmark_count_heuristic.h"
#include "landmarks/landmark_factory_rpg_sasp.h"
#include "lm_cut_heuristic.h"
#include "max_heuristic.h"

#include "operator_cost.h"
#include "search_engine.h"
#include "search_space.h"
#include "state.h"

class TrainingTasks{
public:
    static void solution_path_data_collection(const State &g_initial_state, const int cost_type);
};

#endif
