/*
** derived-preds.c
** 
** Compute the derived predicates for Learning TILDE Trees with derived predicates 
** Date: 2011.09.16
*/

#include "matcher.h"
#include "derived-preds.h"
#include "hornclause-reader.h"

Facts *newfrom_facts(Facts *facts){
  Facts *result = (Facts*) calloc(1, sizeof(Facts));
  result->fact = facts->fact;
  result->index = facts->index;
  result->next = NULL;

  return result;
}


Facts *append_with_facts(Facts *newfact, Facts *list){
  Facts *ifact, *result, *current;
  
  current = (Facts*) calloc (1,sizeof(Facts));
  
  for (ifact = list; ifact != NULL; ifact = ifact->next){
    current->next = newfrom_facts(ifact);
    if (!result)
      result = current->next;
    current = current->next;
  }
  current->next = newfrom_facts(newfact);
  return result;
}


Factunion * cartesian_facts(Facts *facts, Factunion *product){
  Factunion *result, *current, *iProduct;
  Facts *ifacts;
  
  current = (Factunion *) calloc (1, sizeof (Factunion));
  
  if (!product){
    for (iProduct = product; iProduct != NULL; iProduct = iProduct->next){
      for (ifacts = facts; ifacts != NULL; ifacts = ifacts->next){
	current->next = (Factunion *) calloc (1, sizeof (Factunion));
	if (!result)
	  result = current->next;
	current->next->factlist = append_with_facts(ifacts, iProduct->factlist);
	current = current->next;
      }
    }
  }
  else{
    for (ifacts = facts; ifacts != NULL; ifacts = ifacts->next){
      current->next = (Factunion *) calloc (1, sizeof (Factunion));
      if (!result)
	result = current->next;
      current->next->factlist = newfrom_facts(ifacts);
      current = current->next;
    }     
  }
  
  return result;
}

Facts * facts_of_predicate(int f_index){
  Facts *this_facts, *current;
  int i;
  
  //current = ( Facts * ) calloc( 1, sizeof( Facts ) );
  current = NULL;

  if (!gis_added[f_index] && !gis_deleted[f_index]){
    for (i = 0; i < gnum_initial_predicate[f_index]; i++){
      if (current == NULL){
	this_facts = ( Facts * ) calloc( 1, sizeof( Facts ) );
	this_facts->fact = &(ginitial_predicate[f_index][i]);
	this_facts->index = -1; /* there is no grelevant index for static facts */
	current = this_facts;
      }
      else{
	current->next = ( Facts * ) calloc( 1, sizeof( Facts ) );
	current->next->fact = &(ginitial_predicate[f_index][i]);
	current->next->index = -1;
	current = current->next;
      }
    }
  }
  else{
    for(i=0; i< gnum_relevant_facts; i++){
      if (grelevant_facts[i].predicate == f_index){
	if (current == NULL){
	  this_facts = ( Facts * ) calloc( 1, sizeof( Facts ) );
	  this_facts->fact = &(grelevant_facts[i]);
	  this_facts->index = i;
	  current = this_facts;
	}
	else{
	  current->next = ( Facts * ) calloc( 1, sizeof( Facts ) );
	  current->next->fact = &(grelevant_facts[i]);
	  current->next->index = i;
	  current = current->next;
	}
      }
    }
  }
  return this_facts;
}


DPand_literal new_DPand_literal(DPclause *clause, Facts *facts1, Facts *facts2){
  DPand_literal new_dpliteral;
  Fact *facts[DPAND_MAX_LITERALS];
  int i,j, argpos;

  facts[0] =  facts1->fact;
  facts[1] =  facts2->fact;
  
  new_dpliteral.factints[0] = facts1->index;
  new_dpliteral.factints[1] = facts2->index;
  new_dpliteral.args = (int *) calloc(clause->nvars, sizeof (int));
  
  for (i = 0; i < clause->nbodypreds; i++){
    for (j = 0; j < clause->bodypreds[i]->nvars; j++){
      argpos = clause->bodypreds[i]->vars[j].argindex;
      new_dpliteral.args[argpos] = facts[i]->args[j];
    }
  }
  
  return new_dpliteral;
}


DPand_literal new_compound_DPand_literal(DPclause *clause, Facts *facts, DPclause *parent_clause, int iliteral, Bool ordered){
  DPand_literal new_dpliteral;
  int i,j, argpos;

  if (ordered){
    new_dpliteral.factints[0] = facts->index;
    new_dpliteral.factints[1] = iliteral;
  }
  else{
    new_dpliteral.factints[1] = facts->index;
    new_dpliteral.factints[0] = iliteral;
  }

  new_dpliteral.args = (int *) calloc(clause->nvars, sizeof (int));
  
  for (i = 0; i < clause->nbodypreds; i++){
    for (j = 0; j < clause->bodypreds[i]->nvars; j++){
      argpos = clause->bodypreds[i]->vars[j].argindex;
      if (clause->bodypreds[i]->type != PREDICATETYPE_DERIVED)
	new_dpliteral.args[argpos] = facts->fact->args[j];
      else
	new_dpliteral.args[argpos] = parent_clause->and_literals[iliteral].args[j];
    }
  }
  
  return new_dpliteral;
}



void compute_derived_literals (DPclause *and_clause){
  DPmetapred  **mpreds;
  DPbodymatch *this_bodymatch;
  DPand_literal *clause_literals;
  int num_clause_literals;
  int i, j, num_flist, num_dplist, possible_literals;
  int nliterals[DPAND_MAX_LITERALS];
  Facts *ifacts, *jfacts;
  Facts *factlist[DPAND_MAX_LITERALS];
  DPclause *clauselist[DPAND_MAX_LITERALS];
  Bool compound_ordered = TRUE;
  int k_pos1, k_pos2;

  mpreds = and_clause->bodypreds;

  for(i=0; i < and_clause->nbodypreds; i++){
    nliterals[i] = 0;
    
    if (mpreds[i]->type == PREDICATETYPE_DERIVED)
      nliterals[i] = gDP_bgclauses[mpreds[i]->pred_index].num_and_literals;
    else if (!gis_added[mpreds[i]->pred_index] && !gis_deleted[mpreds[i]->pred_index])
      nliterals[i] = gnum_initial_predicate[mpreds[i]->pred_index];
    else{
      for(j=0; j< gnum_relevant_facts; j++){
	if (grelevant_facts[j].predicate == mpreds[i]->pred_index) 
	  nliterals[i]++;
      }
    }
//     printf ("\n >> literals for 1: %d: ", nliterals[i]);
  }

  /*computing the total number of possible and_literals*/
  possible_literals = 1;
  for(i=0; i < and_clause->nbodypreds; i++){
    possible_literals = possible_literals * nliterals[i];
  }
  
  clause_literals = (DPand_literal *) calloc( possible_literals, sizeof(DPand_literal));
  num_clause_literals = 0;
 
 
  num_dplist = 0;
  num_flist = 0;
  for(i=0; i < and_clause->nbodypreds; i++){
    if (mpreds[i]->type != PREDICATETYPE_DERIVED){
      factlist[num_flist++] = facts_of_predicate(mpreds[i]->pred_index);
    }
    else{
      clauselist[num_dplist++] = &(gDP_bgclauses[mpreds[i]->pred_index]);
      if (i==0) compound_ordered = FALSE;
    }
  }
  
  if (compound_ordered)
    {k_pos1 = 0; k_pos2 = 1;}
  else
    {k_pos1 = 1; k_pos2 = 0;}

  if (num_flist == 2) /* both are single predicates*/
    {
      for (ifacts = factlist[0]; ifacts != NULL; ifacts = ifacts->next){
	for (jfacts = factlist[1]; jfacts != NULL; jfacts = jfacts->next){
	  for (this_bodymatch = and_clause->bodymatch; this_bodymatch != NULL; this_bodymatch = this_bodymatch->next){
	    if (ifacts->fact->args[this_bodymatch->argpos[0]] != jfacts->fact->args[this_bodymatch->argpos[1]])
	      break;
	  }
	  if (this_bodymatch == NULL){ /*all matches succeded*/
	    clause_literals[num_clause_literals] = new_DPand_literal(and_clause,ifacts,jfacts);
	    num_clause_literals++;
	  }
	}
      }
    }
  else if (num_flist == 1) /* there is one single predicate and the other is derived (computed previously) */
    {
      for (ifacts = factlist[0]; ifacts != NULL; ifacts = ifacts->next){
	for (j=0; j < clauselist[0]->num_and_literals; j++){
	  for (this_bodymatch = and_clause->bodymatch; this_bodymatch != NULL; this_bodymatch = this_bodymatch->next){
	    if (ifacts->fact->args[this_bodymatch->argpos[k_pos1]] != 
		clauselist[0]->and_literals[j].args[this_bodymatch->argpos[k_pos2]])
	      break;
	  }
	  if (this_bodymatch == NULL){ /*all matches succeded*/
	    clause_literals[num_clause_literals] = new_compound_DPand_literal(and_clause,ifacts,clauselist[0], j, compound_ordered);
	    num_clause_literals++;
	  }
	}
      }
    }
  else 
    {
      printf ("\n\n Why two compound derived?? DEBUGME!!");
      exit(-1);
    }




  
  and_clause->and_literals = clause_literals;
  and_clause->num_and_literals = num_clause_literals;
}



void initialize_DPclause_states(void){
  int i,j,k;
  DPand_literal *dpliterals;

  for (i=0; i< gDP_num_clauses; i++){
    dpliterals = gDP_bgclauses[i].and_literals;
    
    for (j=0; j< gDP_bgclauses[i].num_and_literals; j++){
      dpliterals[j].req_ints = 0;
      for (k=0; k < gDP_bgclauses[i].nbodypreds; k++){
	if (dpliterals[j].factints[k] >= 0)
	  dpliterals[j].req_ints++;
      }
	
    }

    gDP_bgclauses[i].dpstate_type = DPSTATE_STATIC;
    for (j=0; j< gDP_bgclauses[i].nbodypreds; j++){
      if (gDP_bgclauses[i].bodypreds[j]->type == PREDICATETYPE_NEXT ||
	  gDP_bgclauses[i].bodypreds[j]->type == PREDICATETYPE_DERIVED){
	gDP_bgclauses[i].dpstate_type = DPSTATE_NEXT;
	break;
      }
      else if (gDP_bgclauses[i].bodypreds[j]->type != PREDICATETYPE_STATIC && 
	       gDP_bgclauses[i].dpstate_type == DPSTATE_STATIC){
	gDP_bgclauses[i].dpstate_type = DPSTATE_CURRENT;
      }
	
    }
      
  }
}

void reset_DPclause_state(DPclause *clause){
  int j;
  DPand_literal *dpliterals;

  dpliterals = clause->and_literals;
  
  for (j=0; j< clause->num_and_literals; j++){
    dpliterals[j].state_req = dpliterals[j].req_ints;
    dpliterals[j].dp_state = FALSE;
  }
}


void dpstate_from_facts(State *state, DPclause *clause, int mpred_index){
  int i,j;
  int fact_index;
  Fact *fact; 
  DPmetapred *mpred;

  mpred = clause->bodypreds[mpred_index];

  for ( i = 0; i < state->num_F; i++ ){ 
    
    fact_index = state->F[i];
    fact =  &(grelevant_facts[fact_index]);
    
    if (fact->predicate == mpred->pred_index){
      for (j = 0; j < clause->num_and_literals; j++){
	if (clause->and_literals[j].factints[mpred_index] == fact_index)
	  clause->and_literals[j].state_req--;
      }
    }
  }
}



void dpstate_from_goals(State *state, DPclause *clause, int mpred_index){
  int i,j;
  int goal_index;
  Fact *fact; 
  DPmetapred *mpred;

  mpred = clause->bodypreds[mpred_index];

  for ( i = 0; i < gnum_flogic_goal; i++ ){ 
    
    goal_index = gflogic_goal[i];
    fact =  &(grelevant_facts[goal_index]);
    
    if (fact->predicate == mpred->pred_index){
      if (mpred->type == PREDICATETYPE_TARGET && !pending_goal(goal_index))
	continue;
      
      if (mpred->type == PREDICATETYPE_ACHIEVEDGOAL && pending_goal(goal_index))
	continue;
      
      for (j = 0; j < clause->num_and_literals; j++)
	if (clause->and_literals[j].factints[mpred_index] == goal_index)
	  clause->and_literals[j].state_req--;
    }
  }
}


void dpstate_from_derived(State *state, DPclause *clause, int mpred_index){
  int i,j;
  int dplit_index;
  Fact *fact; 
  DPmetapred *mpred;
  DPclause *parent_clause;

  mpred = clause->bodypreds[mpred_index];
  parent_clause = &(gDP_bgclauses[mpred->pred_index]);
  
  for ( i = 0; i < parent_clause->num_and_literals; i++ ){ 
    for (j = 0; j < clause->num_and_literals; j++){
      dplit_index = clause->and_literals[j].factints[mpred_index];
      if (parent_clause->and_literals[dplit_index].dp_state == TRUE)
	clause->and_literals[j].state_req--;
    }
  }
}

  
void compute_dpcurrent_state(State *nextS, int dpstate_type){
  int i, j;
  for (i=0; i< gDP_num_clauses; i++){
    if (gDP_bgclauses[i].dpstate_type == dpstate_type){
      reset_DPclause_state(&(gDP_bgclauses[i]));
      
      for (j = 0; j < gDP_bgclauses[i].nbodypreds; j++){
	switch(gDP_bgclauses[i].bodypreds[j]->type){
	case PREDICATETYPE_STATE: 
	  dpstate_from_facts(LT_current_state, &(gDP_bgclauses[i]),j); break;
	case PREDICATETYPE_NEXT: 
	  dpstate_from_facts(nextS, &(gDP_bgclauses[i]),j); break;
	case PREDICATETYPE_TARGET: 
	case PREDICATETYPE_GOAL:
	case PREDICATETYPE_ACHIEVEDGOAL: 
	  dpstate_from_goals(LT_current_state, &(gDP_bgclauses[i]),j); break;
	}
      }

      for (j = 0; j < gDP_bgclauses[i].num_and_literals; j++){
	if (gDP_bgclauses[i].and_literals[j].state_req == 0)
	  gDP_bgclauses[i].and_literals[j].dp_state = TRUE;
      }
    }
  }
}


void print_derivedstate(int dpstate_type){
  int i, j, k;
  DPclause *clause;

  for (i=0; i< gDP_num_clauses; i++){
    clause = &(gDP_bgclauses[i]);
    
    if (clause->dpstate_type == dpstate_type){
      for (j = 0; j < clause->num_and_literals; j++){
	if (clause->and_literals[j].dp_state){
	  printf("\n %s(", clause->head->name);
	  for (k = 0; k < clause->nvars; k++){
	    if (k>0) 
	      printf(",");
	    printf("%s", gconstants[clause->and_literals[j].args[k]]);
	  }
	  printf(")");
	}
      }
    }

  }
}





