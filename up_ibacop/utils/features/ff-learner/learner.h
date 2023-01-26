/*
** matcher.h
** 
** Made by Raquel Fuentetaja & Tomas de la Rosa
** 
** Started on Mon Oct 20 14:19:48 2008 Raquel
*/

#ifndef   	LEARNER_H_
# define   	LEARNER_H_


#include <strings.h>
#include <ctype.h>
#include "ff.h"

#define PREFERENCE_PARALELISM 1
#define PREFERENCE_DIFFICULTY 2


char * toStr_prolog( char *string);
char * toStr_lower( char *string);
void add_found_solution( BfsNode *last, Bool tag_solution );
void print_solution_path(Solution *sol);
Bool is_target_goal(int fact_index, State *S);

float rank_solution_preference(Solution *iSol, int Type);
int compare_ranked_sols( const void* a, const void* b );
void compute_static_literals();
void tag_sols_commutative_states(void);
Bool commutative_actions(int op1, int op2);
void draw_traverse_tree(FILE *draw_file, BfsNode *node);
void fprint_op_name(FILE *ex_file, int index);

#endif /* _LEARNER_H */


