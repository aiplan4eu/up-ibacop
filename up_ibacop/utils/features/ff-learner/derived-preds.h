/*
** derived-preds.c
** 
** Compute the derived predicates for Learning TILDE Trees with derived predicates 
** Date: 2011.09.16
*/

#ifndef   	DERIVEDPRED_H_
#define   	DERIVEDPRED_H_

#include "ff.h"
#include "output.h"
#include <string.h>



void initialize_DPclause_states(void);
void compute_dpcurrent_state(State *nextS, int dpstate_type);
void print_derivedstate(int dpstate_type);


// void compute_derived_literals(DPclause *and_clause);



#endif 	    /* !DERIVEDPRED_H_ */
