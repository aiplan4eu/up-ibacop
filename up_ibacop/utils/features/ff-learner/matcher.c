/*
** matcher.c
** 
** Made by Raquel Fuentetaja Pizán & Sergio Jiménez Celorrio & Tomas de la Rosa
** Login   <rfuentet@inf.uc3m.es>
** 
** Started on  Mon Oct 20 15:05:58 2008 Raquel
*/


#include "matcher.h"
#include "ff.h"
#include "output.h"
#include "landmarks.h"
#include "hornclause-reader.h"
#include "derived-preds.h"
#include <string.h>
#include "memory.h"


void free_matches(LTmatch *matches_root)
{

  LTmatch *next_match;

  if (matches_root == NULL) return;
  
  next_match = matches_root->next;

  free_bindings(matches_root->bindings);
  free(matches_root);
  free_matches(next_match);
}


 
void free_bindings(int *bindings)
{
  free(bindings);
}


void copy_bindings(LTmatch *match, int *bindings)
{
  int i;

  if (match != NULL)
    {
      for (i = 0; i< LT_num_vars; i++)
	{
	  bindings[i]=match->bindings[i];
	}
      
    }
}


void initialize_matches(LTnode *root)
{
  if (!root) return;

  free_matches(root->matches);

  root->matches = NULL;

  if (root->type != NODETYPE_LEAF)
    {
      initialize_matches(root->node_yes);
      if (root->type != NODETYPE_TAG)
	initialize_matches(root->node_no);
    }

}

void set_tagbounded(LTnode *root)
{
  LTnode *first_node;
  int j;
  Bool tag_bounded = TRUE;



  if (root->type == NODETYPE_TAG)
    {
      first_node = root->node_yes;

      
      /* asking whether the node is helfpul and the father has the extra argument for the class*/
      if (first_node->type == PREDICATETYPE_CANDIDATE &&
 	  root->nvars == first_node->nvars)
	{
	  for (j=0; j< first_node->nvars; j++)
	    {
	      if (first_node->vars[j].free)
		tag_bounded = FALSE;
	    }
	  if (tag_bounded){
	    first_node->tag_bounded = TRUE;
	  }
	}
  
    }
}


void initialize_roller_values(void)
{
  int i;

  for(i = 0; i < gnum_H; i++)
    {
      gop_conn[gH[i]].roller_value = 0;
    }

}


void recompute_target_goals (){
  static Bool first_call = TRUE;
  int i, goal;

  if (first_call)
    {
      gtarget_goal = ( int * ) calloc( gnum_logic_goal, sizeof( int ));
      first_call = FALSE;
    }
  
  gnum_target_goal = 0;

  for(i = 0; i < gnum_flogic_goal; i++)
    {
      goal = gflogic_goal[i]; /* fact index */

      if (pending_goal(goal))
	{
	gtarget_goal[gnum_target_goal++] = goal;
	}
    }
}



int compare_actions_proposal( const void* a, const void* b ) {
   int* arg1 = ( int*) a;
   int* arg2 = ( int*) b;

   float value1= gop_conn[*arg1].roller_value;
   float value2= gop_conn[*arg2].roller_value;

   if (value1 > value2)
     return -1;
   return 1;
}




/* main function */
void match_all(void)
{
  int i, index_bindings_tree;
  struct tms start, end;
  float max_roller_value_gH = 0.0;

  
  times (&start);

  LTnode *nproposal;

  int selected, rejected, op_value;

  
  recompute_target_goals();
  initialize_roller_values();

  /* operators tree*/

  initialize_matches(LT_tree_head);

  /*trace*/
  if ( gcmd_line.display_info == 130 ) {
    print_context();
    printf("\n\n**MATCHING OPERATORS TREE**");
  }

  nproposal=match(LT_tree_head->node_yes);
  selected_proposal = nproposal->proposal;

  /*trace*/
  if ( gcmd_line.display_info == 130 ) {
  printf("\n");
  print_LTproposal(selected_proposal);
  fflush(stdout);
  }


  for (i =0; i< gnum_H; i++)
    {
      op_value = (selected_proposal->elements[extract_operator_from_action(gH[i])]).value ;
      
      /* trace */
      if ( gcmd_line.display_info == 130 ) {
	if (op_value > 0)
	  {
	    printf("\n\n\n**MATCHING BINDINGS TREE FOR: ");
	    print_op_name(gH[i]);
	    fflush(stdout);
	  }
      }
      

      if (op_value > 0)
	{
	  index_bindings_tree = extract_operator_from_action(gH[i]);
          initialize_matches(LT_tree_bindings_head[index_bindings_tree]);
	  nproposal=match_bindings_tree(LT_tree_bindings_head[index_bindings_tree], gH[i]);

	  /*trace*/
	  if ( gcmd_line.display_info == 130 ) {
	    printf("\n");
	    print_LTproposal(nproposal->proposal);
	    fflush(stdout);
	  }

	selected= nproposal->proposal->elements[0].value;
	rejected= nproposal->proposal->elements[1].value;

	if (selected+rejected > 0)
	  gop_conn[gH[i]].roller_value = (float) op_value + ((float)selected/ (float)(selected+rejected));
	else
	  gop_conn[gH[i]].roller_value = (float) op_value;

	if (max_roller_value_gH < gop_conn[gH[i]].roller_value)
	  max_roller_value_gH =  gop_conn[gH[i]].roller_value;

	}
      else
	gop_conn[gH[i]].roller_value = 0.0;
    }
	

  
  /* for non-helpful actions supposed in gA*/
  for (i =0; i< gnum_A; i++)
    {
      op_value = (selected_proposal->elements[extract_operator_from_action(gA[i])]).value ;
      
      if (op_value > 0 && op_value > max_roller_value_gH)
	{

	  /* trace */
	  if ( gcmd_line.display_info == 130 ) 
	  {
	    printf("\n\n\n**NON-HELPFUL ACTION SELECTED (roller_value %f maxHA_roller_value %f)", (float) op_value,max_roller_value_gH );
	    printf("\n**MATCHING BINDINGS TREE FOR: ");
	    print_op_name(gA[i]);
	    fflush(stdout);
	  }

	  index_bindings_tree = extract_operator_from_action(gA[i]);
          initialize_matches(LT_tree_bindings_head[index_bindings_tree]);
	  nproposal=match_bindings_tree(LT_tree_bindings_head[index_bindings_tree], gA[i]);

	  /*trace*/
	  if ( gcmd_line.display_info == 130 ) {
	    printf("\n");
	    print_LTproposal(nproposal->proposal);
	    fflush(stdout);
	  }

	selected= nproposal->proposal->elements[0].value;
	rejected= nproposal->proposal->elements[1].value;

	if (selected+rejected > 0)
	  gop_conn[gA[i]].roller_value = (float) op_value + ((float)selected/ (float)(selected+rejected));
	else
	  gop_conn[gA[i]].roller_value = (float) op_value;

	/* include it as helpful */
	gH[gnum_H++]=gA[i];
	gop_conn[gA[i]].is_in_A = FALSE;
	gop_conn[gA[i]].is_in_H = TRUE;
	}
      else
	gop_conn[gA[i]].roller_value = 0.0;
    }

  qsort(gH, gnum_H, sizeof(int), compare_actions_proposal);
	
  /* trace */  
  if ( gcmd_line.display_info == 130 ) {
  printf("\n\n**FINAL PROPOSAL FOR ACTIONS ORDER**");

  for (i =0; i< gnum_H; i++)
    {
      printf("\n%f, ",gop_conn[gH[i]].roller_value);
      print_op_name(gH[i]);
      fflush(stdout);
		
    }  
  }

  times (&end);
  TIME (matching_time);
  
}

Bool match_tag(LTnode *root, int ha)
{

 int j;
 Action *a;
 int op;
 int *new_bindings = NULL;
 LTmatch *new_match;

  op = ha;

  a =  gop_conn[op].action;

  if (root->nvars != a->num_name_vars)
    {
      printf("\nDEBUGME!!!\n");
      return FALSE;
    }

  new_bindings = (int * ) calloc( LT_num_vars, sizeof( int ) );

 /* match bindings */
 for (j=0; j< root->nvars; j++)
  {
    /* free variable */
    if (root->vars[j].free)
      {
	new_bindings[root->vars[j].index] = a->name_inst_table[j];
	/*  a->name_inst_table[i] is the index of gconstants */
      }
    /* bounded variable */
    else
      {
	if (new_bindings[root->vars[j].index] != a->name_inst_table[j])
	  {
	  printf("\nDEBUGME 2!!!\n");
	  return FALSE;

	  }
      }
  }
		  
		    
  new_match =  (LTmatch * ) calloc( 1, sizeof( LTmatch ) );
	
  new_match->bindings = new_bindings;

  new_match->index = op;

  new_match->next = NULL;

   /*trace*/
   if ( gcmd_line.display_info == 130 ) {
     print_partially_bounded_LTnode(root, new_bindings);
     print_new_variables_values_LTnode(root, new_bindings);
   }


  root->matches = new_match;

  return TRUE;

}


/* main function for dp-trees */
void match_main_dp_tree(Bool with_nonHA)
{
  int i;
  struct tms start, end;
  State next_S;
  
  times (&start);

  LTnode *nproposal;

  int selected, rejected;

  initialize_roller_values();
  make_state( &next_S, gnum_ft_conn, gnum_fl_conn );

  compute_dpcurrent_state(NULL, DPSTATE_CURRENT);

  if ( gcmd_line.display_info == 130 ) {
    //print_context();
    printf("\n\n**MATCHING DERIVED PREDICATE TREE**\n");
    printf("\n Derived State:");
    print_derivedstate(DPSTATE_CURRENT);
  }
  

  
  /* matching helpful actions */
  for (i =0; i< gnum_H; i++){
    initialize_matches(LT_tree_head);
    
    /*trace*/
    if ( gcmd_line.display_info == 130 ) {
      printf ("\n >> Match for:");
      print_op_name(gH[i]);
    }
    
    if ( result_to_dest( &next_S, LT_current_state, gH[i] ) ) {

      compute_dpcurrent_state(&next_S, DPSTATE_NEXT);

      if ( gcmd_line.display_info == 130 ) {
	printf("\n Next State:\n");
	print_State(next_S);
	printf("\n Derived State:");
	print_derivedstate(DPSTATE_NEXT);
      }
      
      
      nproposal=match_ahead(LT_tree_head, &next_S);
      
      selected= nproposal->proposal->elements[0].value;
      rejected= nproposal->proposal->elements[1].value;

      /*trace*/
      if ( gcmd_line.display_info == 130 ) {
	printf("\n");
	print_LTproposal(nproposal->proposal);
	printf("\n");
	fflush(stdout);
      }

      
      if (selected+rejected > 0)
	gop_conn[gH[i]].roller_value = ((float)selected/ (float)(selected+rejected));
      else
	gop_conn[gH[i]].roller_value = 0.0;
      
    }
  }
  
  /* matching non helpful actions */
  if (with_nonHA){
    for (i =0; i< gnum_A; i++){
      initialize_matches(LT_tree_head);
      
      /*trace*/
      if ( gcmd_line.display_info == 130 ) {
	printf ("\n >> Match Non HA for:");
	print_op_name(gA[i]);
      }
      
      if ( result_to_dest( &next_S, LT_current_state, gA[i] )) {

	compute_dpcurrent_state(&next_S, DPSTATE_NEXT);

	if ( gcmd_line.display_info == 130 ) {
	  printf("\n Next State:\n");
	  print_State(next_S);
	}
	
	nproposal=match_ahead(LT_tree_head, &next_S);
	selected = nproposal->proposal->elements[0].value;
	rejected = nproposal->proposal->elements[1].value;
	
	/* Include the nonH haction in gH if is a selected leaf */
	if (selected > rejected){
	/*trace*/
	  if ( gcmd_line.display_info == 130 ) {
	    printf("\n");
	    print_LTproposal(nproposal->proposal);
	    printf("\n");
	    fflush(stdout);
	  }
	  
	  gop_conn[gA[i]].roller_value = ((float)selected/ (float)(selected+rejected));

	  gH[gnum_H++]=gA[i];
	  gop_conn[gA[i]].is_in_A = FALSE;
	  gop_conn[gA[i]].is_in_H = TRUE;
	  
	}
	  
	else{
	  gop_conn[gA[i]].roller_value = 0.0;

	  if ( gcmd_line.display_info == 130 ) {
	    printf("\n > Non_HA discarded ");
	  }
	  
	}
      }
      
    }
    
  }
  
  qsort(gH, gnum_H, sizeof(int), compare_actions_proposal);
  
  /* trace */  
  if ( gcmd_line.display_info == 130 ) {
    printf("\n\n**FINAL PROPOSAL FOR ACTIONS ORDER**");
    
    for (i =0; i< gnum_H; i++)
      {
	printf("\n%f, ",gop_conn[gH[i]].roller_value);
	print_op_name(gH[i]);
	fflush(stdout);
	
      }  
  }
  
  times (&end);
  TIME (matching_time);
}


/* This makes the match with the candidate states
   Note that in Roller match, we use the current state */


LTnode* match_ahead(LTnode *root, State *nextS)
{
   
  if (root->type == NODETYPE_LEAF)
    return root;

  compute_matches(root, nextS);

  if (root->matches !=NULL)
    {
      if(root->node_yes->type == NODETYPE_LEAF)
	return root->node_yes;
      else
	{
	  return match_ahead(root->node_yes, nextS);
	}
    }
  else
    {
      return match_ahead(root->node_no, nextS);
    }
  
}





LTnode* match(LTnode *root)
{
   
  if (root->type == NODETYPE_LEAF)
    return root;

  compute_matches(root, NULL);

  if (root->matches !=NULL)
    {
      /* trace */
      /*
	if ( gcmd_line.display_info == 130 ) {
	printf("\n\nROLLER: MATCH ");
	print_LTtype(root->type);
	fprintf(stdout,"%s\n", root->pred);
	fflush(stdout);
	}
      */

      if(root->node_yes->type == NODETYPE_LEAF)
	return root->node_yes;

      else
	{
	  return match(root->node_yes);
	}
    }
  
  else
    {
      /*trace*/
      /*
	if ( gcmd_line.display_info == 130 ) {
	printf("\n\nROLLER: NO MATCH ");
	print_LTtype(root->type);
	fprintf(stdout,"%s\n", root->pred);
	fflush(stdout);
	}
      */
      return match(root->node_no);
    }
  
}





LTnode* match_bindings_tree(LTnode *root, int ha)
{
 
  if (root->type == NODETYPE_TAG)
    {
      match_tag(root, ha);
      return  match(root->node_yes);
    }
  else if (root->type == NODETYPE_LEAF)
    {
      return match(root);
    }
  else
    {
      printf("\nDEBUGME!!! strange bindings tree to match\n");
      return FALSE;
    }

}


LTnode* get_last_match_father(LTnode *current)
{
  LTnode *last_father;
  
  last_father = current->father;
  while (last_father)
    {
      if (last_father->matches)
	return last_father;
      else
	last_father = last_father->father;
    }
  return NULL;
}



Bool match_candidate_bounded_tag(LTnode *root)
{ 
  int i,j,l;
  Action *a;
  int op;
  int *new_bindings = NULL;
  LTmatch *new_match;
  LTmatch *last_match;
  Bool there_is_match = FALSE;
  Bool there_is_new_match = FALSE;
  LTnode *last_father;
  
  /* searching for previous fathers matches*/
  last_father = root->father;
  /*actual_father_match = last_father->matches;*/
  
  
  /*trace*/
  /*if ( gcmd_line.display_info == 130 ) {
    printf("\n");
    print_LTnode(root);
    if (actual_father_match)
      print_partially_bounded_LTnode(root, actual_father_match->bindings);
      }*/
      
  op = last_father->matches->index;
  
  for(i = 0; i < gnum_H; i++)
    {
      if (op == gH[i]){
	there_is_match = TRUE;
	break;
      }
      
    }
  
  if (there_is_match)
    {
      a = gop_conn[op].action;
      
      /* at this point there is a match with the father */
      new_bindings = (int * ) calloc( LT_num_vars, sizeof( int ) );
      copy_bindings(last_father->matches, new_bindings);
      
      new_match =  (LTmatch * ) calloc( 1, sizeof( LTmatch ) );
      new_match->bindings = new_bindings;
      new_match->index = op;
      new_match->next = NULL;
      
      /*trace*/
      if ( gcmd_line.display_info == 130 ) {
	print_new_variables_values_LTnode(root, new_bindings);
	
      }
      
      if (root->matches == NULL)
	{
	  root->matches = new_match;
	}
      else
	{
	  printf ("DEBUG ME!....");
	}
    }

  return there_is_match;
}



Bool match_candidate_ha(LTnode *root)
{ 
  int i,j,l;
  Action *a;
  int op;
  int *new_bindings = NULL;
  LTmatch *actual_father_match = NULL;
  LTmatch *new_match;
  LTmatch *last_match;
  Bool there_is_match = FALSE;
  Bool there_is_new_match = FALSE;
  Bool tree_head = FALSE;
  LTnode *last_father;
  
  
  
  /* searching for previous fathers matches*/
  last_father = root->father;
  while (last_father)
    {
      if (last_father->matches){
	actual_father_match = last_father->matches;
	/*
	  if ( gcmd_line.display_info == 130 ) {
	  printf("\nFather matches");
	  print_matches(last_father);
	  }
	*/
	
	break;
      }
      else
	{
	  last_father= last_father->father;
	}
      
    }
  
  if (!last_father)
    tree_head = TRUE;
  
  
  while(actual_father_match || tree_head)
    {
      /*trace*/
      if ( gcmd_line.display_info == 130 ) {
	printf("\n");
	print_LTnode(root);
	if (actual_father_match)
	  print_partially_bounded_LTnode(root, actual_father_match->bindings);
      }
      
      
      for(i = 0; i < gnum_H; i++)
	{
	  op = gH[i];
	  
	  a = gop_conn[op].action;
	  
	  /* same action name */
	  //	if (strcasecmp (a->name, root->pred) == SAME)
	  if (a->name_index == root->pred_index)
	    {
	      
	      if (root->nvars == a->num_name_vars)
		{
		  /* checking whether the bounded variables match with the father node */
		  if (last_father)
		    {
		      for (j=0; j< root->nvars; j++)
			{
			  if (!root->vars[j].free)
			    {
			      if (actual_father_match->bindings[root->vars[j].index] != a->name_inst_table[j])
				break;
			    }
			}
		      /* there is no match*/
		      if (j != root->nvars)
			{
//  			  printf("\n continuing breaking match..candidate_ha");
			  continue;
			}
		    }

		  /* at this point there is a match with the father */
		  new_bindings = (int * ) calloc( LT_num_vars, sizeof( int ) );

		  if (last_father)
		    {
		      copy_bindings(actual_father_match, new_bindings);
		    }
		  
		  /* adding free variables */
		  for (j=0; j< root->nvars; j++)
		    {
		      if (root->vars[j].free)
			{
			  new_bindings[root->vars[j].index] = a->name_inst_table[j];
			}
		    }
		  
		  there_is_match = TRUE;
		  there_is_new_match = TRUE;
		      
		  new_match =  (LTmatch * ) calloc( 1, sizeof( LTmatch ) );
		     
		  new_match->bindings = new_bindings;
		  new_match->index = op;
		  new_match->next = NULL;
		      
		  /*trace*/
		  if ( gcmd_line.display_info == 130 ) {
		    print_new_variables_values_LTnode(root, new_bindings);
		    
		  }
		      
		  if (root->matches == NULL)
		    {
		      root->matches = new_match;
		      last_match = root->matches;
		      
		    }
		  else
		    {
		      last_match->next = new_match;
		      last_match = last_match->next;
		    }
		}
	    }
	}
      
      if (tree_head)
	break;
      
      /* compute next binding only if different values for the root variables */
      while (actual_father_match->next)
	{
	  for(l = 0; l < root->nvars; l++)
	    {
	      if (!root->vars[l].free )
		{
		  if (actual_father_match->bindings[root->vars[l].index]!= 
		      actual_father_match->next->bindings[root->vars[l].index])
		    break;
		}
	 }
	  
	  /* Same values for actual_father_match->next*/
	  if (l == root->nvars)
	    {
	      actual_father_match = actual_father_match->next;
	    }
	  else
	    break;
	}
      
      
      actual_father_match = actual_father_match->next;
      
      
      /* trace */
      if ( gcmd_line.display_info == 130 ) 
	{
	  if (!there_is_new_match)
	    printf("\n?- No");
	  if (actual_father_match)
	    printf("\n...Redo ");
	  there_is_new_match = FALSE;
	}
    }
  
  if ( gcmd_line.display_info == 130 ) {
    if (!there_is_match){
      printf("\n?- No (node)");
    }
  }
  
  return there_is_match;
}


Bool match_target_goal(LTnode *root)
{
  int i,j,l;
 Fact *fact;
 int goal;
 int *new_bindings = NULL;
 LTmatch *actual_father_match = NULL;
 LTmatch *new_match;
 LTmatch *last_match;
 Bool there_is_match = FALSE;
 Bool there_is_new_match =FALSE;
 Bool tree_head = FALSE;
 LTnode *last_father;
  

  /* searching for previous fathers matches*/
 last_father = root->father;
 while (last_father)
   {
     if (last_father->matches){
       actual_father_match = last_father->matches;
       /*
       if ( gcmd_line.display_info == 130 ) {
	 printf("\nFather matches");
	 print_matches(last_father);
       }
       */

       break;
     }
     else
       {
	 last_father= last_father->father;
       }

   }

 if (!last_father)
     tree_head = TRUE;


 while(actual_father_match || tree_head)
   {
     /*trace*/
     if ( gcmd_line.display_info == 130 ) {
       printf("\n");
       print_LTnode(root);
       if (actual_father_match)
	 print_partially_bounded_LTnode(root, actual_father_match->bindings);
     }

   for(i = 0; i < gnum_target_goal; i++)
      {
	goal = gtarget_goal[i]; /* fact index */

	fact =  &(grelevant_facts[goal]);

	/* same predicate name */
	if (fact->predicate == root->pred_index)
	  {
 	    if (root->nvars == garity[fact->predicate])
 	      {
		/* checking whether the bounded variables match with the father node */
		  if (last_father)
		    {
		      for (j=0; j< root->nvars; j++)
			{
			  if (!root->vars[j].free)
			    {
			      if (actual_father_match->bindings[root->vars[j].index] != (fact->args)[j])
				break;
			    }
			}
		      /* there is no match*/
		      if (j != root->nvars)
			{
// 			  printf("\n continuing breaking match GOALS...");
			  continue;
			}
		    }

		  
		  /* at this point there is a match with the father */
		  new_bindings = (int * ) calloc( LT_num_vars, sizeof( int ) );
		
		  if (last_father )
		    {
		      copy_bindings(actual_father_match, new_bindings);
		    }
		  
		  /* match bindings */
		  for (j=0; j< root->nvars; j++)
		    {
		      /* free variable */
		      if (root->vars[j].free)
		      {
			new_bindings[root->vars[j].index] = (fact->args)[j];
		      }

		    }
		  
		  /* match found */
		  there_is_match = TRUE;
		  there_is_new_match = TRUE;
		  
		  new_match =  (LTmatch * ) calloc( 1, sizeof( LTmatch ) );
		  
		  new_match->bindings = new_bindings;
		  
		  new_match->next = NULL;
		  
		  new_match->index = goal;
		  
		  
		  /*trace*/
		  if ( gcmd_line.display_info == 130 ) {
		    print_new_variables_values_LTnode(root, new_bindings);
		  }
		  
		  
		  if (root->matches == NULL)
		    {
		      
		      root->matches = new_match;
		      last_match = root->matches;
		      
		    }
		  else
		    {
		      last_match->next = new_match;
		      last_match = last_match->next;
		    }
		  
	      }
	  }
	
      }

   if (tree_head)
     break;
   
   /* compute next binding only if different values for the root variables */
   while (actual_father_match->next)
     {
       for(l = 0; l < root->nvars; l++)
	 {
	   if (!root->vars[l].free )
	     {
	       if (actual_father_match->bindings[root->vars[l].index]!= 
		   actual_father_match->next->bindings[root->vars[l].index])
		 break;
	     }
	 }
       
       /* Same values for actual_father_match->next*/
       if (l == root->nvars)
	 {
	   actual_father_match = actual_father_match->next;
	 }
       else
	 break;
     }
	 
   
   actual_father_match = actual_father_match->next;
   
   
   /* trace */
   if ( gcmd_line.display_info == 130 ) 
     {
       if (!there_is_new_match)
	 printf("\n?- No");
       if (actual_father_match)
	 printf("\n...Redo ");
       there_is_new_match = FALSE;
     }
   }
 
 if ( gcmd_line.display_info == 130 ) {
   if (!there_is_match){
     printf("\n?- No (node)");
   }
 }
 
 return there_is_match;
}

     
Bool match_static(LTnode *root)
{
  int i,j,k,l;
  Fact *fact;
  int *new_bindings = NULL;
  LTmatch *actual_father_match = NULL;
  LTmatch *new_match;
  LTmatch *last_match;
  Bool there_is_match = FALSE;
  Bool there_is_new_match = FALSE;
  Bool tree_head = FALSE;
  LTnode *last_father;
  
  
  /* searching for previous fathers matches*/
  last_father = root->father;
  while (last_father)
    {
      if (last_father->matches){
	actual_father_match = last_father->matches;
	/*
	  if ( gcmd_line.display_info == 130 ) {
	  printf("\nFather matches");
	  print_matches(last_father);
	  }
	*/
	
	break;
      }
      else
	{
	  last_father= last_father->father;
	}
    }
  
  if (!last_father)
    tree_head = TRUE;

  
  while(actual_father_match || tree_head)
    {
      /*trace*/
      if ( gcmd_line.display_info == 130 ) {
	printf("\n");
	print_LTnode(root);
	if (actual_father_match)
	  print_partially_bounded_LTnode(root, actual_father_match->bindings);
      }
      
      for(i = 1; i < gnum_orig_predicates; i++)
	{
	  if (!gis_added[i] && !gis_deleted[i])
	    /* static predicate */
	    {
	      for ( k = 0; k < gnum_initial_predicate[i]; k++ ) 
		{
		  fact = &(ginitial_predicate[i][k]) ; 
		  
		  /* same predicate name */
		  if (fact->predicate == root->pred_index)
		    {
		      if (root->nvars == garity[fact->predicate])
			{
			  /* checking whether the bounded variables match with the father node */
			  if (last_father)
			    {
			      for (j=0; j< root->nvars; j++)
				{
				  if (!root->vars[j].free)
				    {
				      if (actual_father_match->bindings[root->vars[j].index] != (fact->args)[j])
					break;
				    }
				}
			      /* there is no match*/
			      if (j != root->nvars)
				{
// 				  printf("\n continuing breaking match...static");
				  continue;
				}
			    }

			  /* at this point there is a match with the father */
			  new_bindings = (int * ) calloc( LT_num_vars, sizeof( int ) );
			  
			  /* copy father bindings */
			  if (last_father )
			    {
			      copy_bindings(actual_father_match, new_bindings);
			    }
			  
			  /* match bindings */
			  for (j=0; j< root->nvars; j++)
			    {
			      if (root->vars[j].free)
				{
				  new_bindings[root->vars[j].index] = (fact->args)[j];
				}
			    }
			  
			  there_is_match = TRUE;
			  there_is_new_match = TRUE;
			  
			  new_match =  (LTmatch * ) calloc( 1, sizeof( LTmatch ) );
			  
			  new_match->bindings = new_bindings;
			  new_match->next = NULL;
			      
			  /*trace*/
			  if ( gcmd_line.display_info == 130 ) {
			    print_new_variables_values_LTnode(root, new_bindings);
			  }
			  
			  if (root->matches == NULL)
			    {
			      root->matches = new_match;
			      last_match = root->matches;
			      
			    }
			  else
			    {
			      last_match->next = new_match;
			      last_match = last_match->next;
			    }
			}
		    }
		}
	    }
	}
      
      
      if (tree_head)
	break;
      
      /* compute next binding only if different values for the root variables */
      while (actual_father_match->next)
	{
	  for(l = 0; l < root->nvars; l++)
	    {
	      if (!root->vars[l].free )
		{
		  if (actual_father_match->bindings[root->vars[l].index]!= 
		      actual_father_match->next->bindings[root->vars[l].index])
		    break;
		}
	    }
	  
	  /* Same values for actual_father_match->next*/
	  if (l== root->nvars)
	    {
	      actual_father_match = actual_father_match->next;
	    }
	  else
	    break;
	}
      
      
      
      actual_father_match = actual_father_match->next;
      
      /* trace */
      if ( gcmd_line.display_info == 130 ) 
	{
	  if (!there_is_new_match)
	    printf("\n?- No");
	  if (actual_father_match)
	    printf("\n...Redo ");
	  there_is_new_match = FALSE;
	}
    }
  
  if ( gcmd_line.display_info == 130 ) {
    if (!there_is_match){
      printf("\n?- No (node)");
    }
  }
  
  return there_is_match;
}


Bool match_executed(LTnode *root)
{
  int j,l;
 Action *a;
 int *new_bindings = NULL;
 LTmatch *actual_father_match = NULL;
 LTmatch *new_match;
 LTmatch *last_match;
 Bool there_is_match = FALSE;
 Bool there_is_new_match = FALSE;
 Bool tree_head = FALSE;
 LTnode *last_father;
 
 if (!LT_last_executed_action) return FALSE;
  
 last_father = get_last_match_father(root);
 
 if (last_father)
   actual_father_match = last_father->matches;
 else
   tree_head = TRUE;
 
 while(actual_father_match || tree_head)
   {

     /*trace*/
     if ( gcmd_line.display_info == 130 ) {
       printf("\n");
       print_LTnode(root);
       if (actual_father_match)
	 print_partially_bounded_LTnode(root, actual_father_match->bindings);
     }

  
    a = LT_last_executed_action;

    /* same action name */
    //    if (strcasecmp (a->name, root->pred) == SAME)
    if (a->name_index == root->pred_index)
      {

	if (root->nvars == a->num_name_vars)
	  {
	    
	    new_bindings = (int * ) calloc( LT_num_vars, sizeof( int ) );
	    
	    /* copy father bindings */
	    if (last_father)
	      {
		copy_bindings(actual_father_match, new_bindings);
	      }
	    
	    /* match bindings */
	    for (j=0; j< root->nvars; j++)
	      {
		/* free variable */
		if (root->vars[j].free)
		  {
		    new_bindings[root->vars[j].index] = a->name_inst_table[j];		
		    /*  a->name_inst_table[i] is the index of gconstants */
		    
		  }
		
		/* bounded variable */
		else
		  {
		    if (new_bindings[root->vars[j].index] !=  a->name_inst_table[j])
		      break;
		  }
	      }
	    
	    /* there is no match*/
	    if (j != root->nvars)
	      {
		free_bindings(new_bindings);
	      }
	    else
	      {
		/* match found */
		there_is_match = TRUE;
		there_is_new_match = TRUE;
		
		new_match =  (LTmatch * ) calloc( 1, sizeof( LTmatch ) );
		
		new_match->bindings = new_bindings;
		new_match->index = LT_last_executed_index;
		new_match->next = NULL;
		
		/*trace*/
		if ( gcmd_line.display_info == 130 ) {
		  print_new_variables_values_LTnode(root, new_bindings);
		}
		
		if (root->matches == NULL)
		  {
		    root->matches = new_match;
		    last_match = root->matches;
		  }
		else
		  {
		    last_match->next = new_match;
		    last_match = last_match->next;	
		  }
	      }
	  }
      }
    
    if (tree_head)
     break;
   
   /* compute next binding only if different values for the root variables */
   while (actual_father_match->next)
     {
       for(l = 0; l < root->nvars; l++)
	 {
	   if (!root->vars[l].free )
	     {
	       if (actual_father_match->bindings[root->vars[l].index]!= 
		   actual_father_match->next->bindings[root->vars[l].index])
		 break;
	     }
	 }
       
       /* Same values for actual_father_match->next*/
       if (l == root->nvars)
	 {
	   actual_father_match = actual_father_match->next;
	 }
       else
	 break;
     }

   actual_father_match = actual_father_match->next;
   
   /* trace */
   if ( gcmd_line.display_info == 130 && actual_father_match )
     {
     if (!there_is_new_match)
       printf("\n?- No");
     if (actual_father_match)
       printf("\n...Redo ");
     there_is_new_match = FALSE;
     }

   }

 if ( gcmd_line.display_info == 130 ) {
   if (!there_is_match){
     printf("\n?- No (node)");
   }
 }
 return there_is_match;
}



Bool match_candidates_ha_and_nonh(LTnode *root)
{
  int i,j,l;
  Action *a;
  int op, a_num;
  int *action_ints;
  int *new_bindings = NULL;
  LTmatch *actual_father_match = NULL;
  LTmatch *new_match;
  LTmatch *last_match;
  Bool there_is_match = FALSE;
  Bool there_is_new_match = FALSE;
  Bool tree_head = FALSE;
  LTnode *last_father;
 
  last_father = get_last_match_father(root);
  
  if (last_father)
    actual_father_match = last_father->matches;
  else
    tree_head = TRUE;
  
  if (root->type == PREDICATETYPE_CANDIDATE){
    a_num = gnum_H;
    action_ints = gH;
  }
  if (root->type == PREDICATETYPE_NOTHELPFUL){
    a_num = gnum_A;
    action_ints = gA;
  }
    
  while(actual_father_match || tree_head)
    {
      /*trace*/
      if ( gcmd_line.display_info == 130 ) {
	printf("\n");
	print_LTnode(root);
	if (actual_father_match)
	  print_partially_bounded_LTnode(root, actual_father_match->bindings);
      }
      

      for(i = 0; i < a_num; i++)
	{
	  op = action_ints[i];
	  a = gop_conn[op].action;
	  
	  /* same action name */
	  //	if (strcasecmp (a->name, root->pred) == SAME)
	  if (a->name_index == root->pred_index)
	    {
	      
	      if (root->nvars == a->num_name_vars)
		{
		  /* checking whether the bounded variables match with the father node */
		  if (last_father)
		    {
		      for (j=0; j< root->nvars; j++)
			{
			  if (!root->vars[j].free)
			    {
			      if (actual_father_match->bindings[root->vars[j].index] != a->name_inst_table[j])
				break;
			    }
			}
		      /* there is no match*/
		      if (j != root->nvars)
			{
			  continue;
			}
		    }
		  
		  /* at this point there is a match with the father */
		  new_bindings = (int * ) calloc( LT_num_vars, sizeof( int ) );
		  
		  /* copy father bindings */
		  if (last_father)
		    {
		      copy_bindings(actual_father_match, new_bindings);
		    }
		  
		  /* adding free variables */
		  for (j=0; j< root->nvars; j++)
		    {
		      if (root->vars[j].free)
			{
			  new_bindings[root->vars[j].index] = a->name_inst_table[j];
			}
		      
		    }
		  
		  there_is_match = TRUE;
		  there_is_new_match = TRUE;
		  
		  new_match =  (LTmatch * ) calloc( 1, sizeof( LTmatch ) );
		  
		  new_match->bindings = new_bindings;
		  new_match->index = op;
		  new_match->next = NULL;
		  
		  /*trace*/
		  if ( gcmd_line.display_info == 130 ) {
		    print_new_variables_values_LTnode(root, new_bindings);
		  }
		  
		  if (root->matches == NULL)
		    {
		      root->matches = new_match;
		      last_match = root->matches;
		    }
		  else
		    {
		      last_match->next = new_match;
		      last_match = last_match->next;
		    }
		}
	    }
	}
      
      if (tree_head)
	break;
      
      /* compute next binding only if different values for the root variables */
      while (actual_father_match->next)
	{
	  for(l = 0; l < root->nvars; l++)
	    {
	      if (!root->vars[l].free )
		{
		  if (actual_father_match->bindings[root->vars[l].index]!= 
		      actual_father_match->next->bindings[root->vars[l].index])
		    break;
		}
	 }
	  
	  /* Same values for actual_father_match->next*/
	  if (l == root->nvars)
	    {
	      actual_father_match = actual_father_match->next;
	    }
	  else
	    break;
	}
      
      
      actual_father_match = actual_father_match->next;
      
      
      /* trace */
      if ( gcmd_line.display_info == 130 ) 
	{
	  if (!there_is_new_match)
	    printf("\n?- No");
	  if (actual_father_match)
	    printf("\n...Redo ");
	  there_is_new_match = FALSE;
	}
    }
  
  if ( gcmd_line.display_info == 130 ) {
    if (!there_is_match){
      printf("\n?- No (node)");
    }
  }
  
  return there_is_match;
  
}

Bool match_goals_target_and_achieved(LTnode *root)
{
  int i,j,l;
  Fact *fact;
  int goal;
  int *new_bindings = NULL;
  LTmatch *actual_father_match = NULL;
  LTmatch *new_match;
  LTmatch *last_match;
  Bool there_is_match = FALSE;
  Bool there_is_new_match =FALSE;
  Bool tree_head = FALSE;
  LTnode *last_father;
  
  
  /* searching for previous fathers matches*/
  last_father = get_last_match_father(root);

  if (last_father)
    actual_father_match = last_father->matches;
  else
    tree_head = TRUE;
  
  
  while(actual_father_match || tree_head)
    {
      /*trace*/
      if ( gcmd_line.display_info == 130 ) {
	printf("\n");
	print_LTnode(root);
	if (actual_father_match)
	  print_partially_bounded_LTnode(root, actual_father_match->bindings);
      }
      
      for(i = 0; i < gnum_flogic_goal; i++)
	{
	  goal = gflogic_goal[i]; /* fact index */
	  
	  /* Matching target and goal in current state, not pending, then continue */
	  if (root->type == PREDICATETYPE_TARGET && !pending_goal(goal))
	    continue;
	  /* Matching achieved and goal not in current state, then continue */
	  if (root->type == PREDICATETYPE_ACHIEVEDGOAL && pending_goal(goal))
	    continue;
	  
	  fact =  &(grelevant_facts[goal]);
	  
	  /* same predicate name */
	  if (fact->predicate == root->pred_index)
	    {
	      if (root->nvars == garity[fact->predicate])
		{
		  new_bindings = (int * ) calloc( LT_num_vars, sizeof( int ) );
		  
		  /* copy father bindings */
		  if (last_father)
		    {
		      copy_bindings(actual_father_match, new_bindings);
		    }
		  
		  /* match bindings */
		  for (j=0; j< root->nvars; j++)
		    {
		      /* free variable */
		      if (root->vars[j].free)
			{
			  
			  new_bindings[root->vars[j].index] = (fact->args)[j];
			  /*  (fact->args)[j] is the index of gconstants */
			  
			}
		      
		      /* bounded variable */
		      else
			{
			  if (new_bindings[root->vars[j].index] != (fact->args)[j])
			    break;
			}
		    }
		  
		  /* there is no match*/
		  if (j != root->nvars)
		    {
		      free_bindings(new_bindings);
		      continue;
		    }
		  else
		    {
		      /* match found */
		      there_is_match = TRUE;
		      there_is_new_match = TRUE;
		      
		      new_match =  (LTmatch * ) calloc( 1, sizeof( LTmatch ) );
     		      new_match->bindings = new_bindings;
		      new_match->next = NULL;
		      new_match->index = goal;
		      
		      /*trace*/
		      if ( gcmd_line.display_info == 130 ) {
			print_new_variables_values_LTnode(root, new_bindings);
		    }
		      
		      
		      if (root->matches == NULL)
			{
			  
			  root->matches = new_match;
			  last_match = root->matches;
			  
			}
		      else
			{
			  last_match->next = new_match;
			  last_match = last_match->next;
			}
		      /*trace*/
		      /*
			if ( gcmd_line.display_info == 130 ) {
			print_matches(root);
			}
		      */
		      
		    }
		}
	    }
	}
      
      if (tree_head)
	break;
      
      /* compute next binding only if different values for the root variables */
      while (actual_father_match->next)
	{
	  for(l = 0; l < root->nvars; l++)
	    {
	      if (!root->vars[l].free )
		{
		  if (actual_father_match->bindings[root->vars[l].index]!= 
		      actual_father_match->next->bindings[root->vars[l].index])
		    break;
		}
	    }
	  
	  /* Same values for actual_father_match->next*/
	  if (l == root->nvars)
	    {
	      actual_father_match = actual_father_match->next;
	    }
	  else
	    break;
	}
      
      
      actual_father_match = actual_father_match->next;
      
      
      /* trace */
      if ( gcmd_line.display_info == 130 ) 
	{
	  if (!there_is_new_match)
	    printf("\n?- No");
	  if (actual_father_match)
	    printf("\n...Redo ");
	  there_is_new_match = FALSE;
	}
    }
  
  if ( gcmd_line.display_info == 130 ) {
    if (!there_is_match){
      printf("\n?- No (node)");
    }
  }
  
  return there_is_match;
}


Bool rpaction_is_helpful(int op){
  int i;

  for (i=0;i<gnum_H;i++){
    if (op == gH[i])
      return TRUE;
  }
  return FALSE;

}


Bool match_rpaction(LTnode *root)
{
  int i,j,l;
  Action *a;
  int op;
  int *new_bindings = NULL;
  LTmatch *actual_father_match = NULL;
  LTmatch *new_match;
  LTmatch *last_match;
  Bool there_is_match = FALSE;
  Bool there_is_new_match = FALSE;
  Bool tree_head = FALSE;
  LTnode *last_father;
  
  /* searching for previous fathers matches*/
  last_father = get_last_match_father(root);

  if (last_father)
    actual_father_match = last_father->matches;
  else
    tree_head = TRUE;
  
  while(actual_father_match || tree_head)
    {
      /*trace*/
      if ( gcmd_line.display_info == 130 ) {
	printf("\n");
	print_LTnode(root);
	if (actual_father_match)
	  print_partially_bounded_LTnode(root, actual_father_match->bindings);
      }
      
      
      for(i = 0; i < LT_current_num_RP; i++)
	{
	  op = LT_current_RP[i];
	  
	  /* we only match rp_actions that are not helpful, so level 2 and so forth */
	  if (rpaction_is_helpful(op))
	    continue;
	  
	  a = gop_conn[op].action;
	  
	  /* same action name */
	  //	if (strcasecmp (a->name, root->pred) == SAME)
	  if (a->name_index == root->pred_index)
	    {
	      
	      if (root->nvars == a->num_name_vars)
		{
		  
		  new_bindings = (int * ) calloc( LT_num_vars, sizeof( int ) );
		  
		  /* copy father bindings */
		  if (last_father)
		    {
		      copy_bindings(actual_father_match, new_bindings);
		    }
		  
		  /* match bindings */
		  for (j=0; j< root->nvars; j++)
		    {
		      /* free variable */
		      if (root->vars[j].free)
			{
			  new_bindings[root->vars[j].index] = a->name_inst_table[j];
			  /*  a->name_inst_table[i] is the index of gconstants */
			}
		      
		      /* bounded variable */
		      else
			{
			  if (new_bindings[root->vars[j].index] != a->name_inst_table[j])
			    break;
			}
		    }
		  
		  /* there is no match*/
		  if (j != root->nvars)
		    {
		      free_bindings(new_bindings);
		      continue;
		    }
		  else
		    {
		      /* match found */
		      there_is_match = TRUE;
		      there_is_new_match = TRUE;
		      
		      new_match =  (LTmatch * ) calloc( 1, sizeof( LTmatch ) );
		      
		      new_match->bindings = new_bindings;
		      new_match->index = op;
		      new_match->next = NULL;
		      
		      /*trace*/
		      if ( gcmd_line.display_info == 130 ) {
			print_new_variables_values_LTnode(root, new_bindings);
			
		      }
		      
		      if (root->matches == NULL)
			{
			  root->matches = new_match;
			  last_match = root->matches;
			  
			}
		      else
			{
			  last_match->next = new_match;
			  last_match = last_match->next;
			  
			}
		      
		    }
		}
	    }
	}
      
      if (tree_head)
	break;
      
      /* compute next binding only if different values for the root variables */
      while (actual_father_match->next)
	{
	  for(l = 0; l < root->nvars; l++)
	    {
	      if (!root->vars[l].free )
		{
		  if (actual_father_match->bindings[root->vars[l].index]!= 
		      actual_father_match->next->bindings[root->vars[l].index])
		    break;
		}
	 }
	  
	  /* Same values for actual_father_match->next*/
	  if (l == root->nvars)
	    {
	      actual_father_match = actual_father_match->next;
	    }
	  else
	    break;
	}
      
      
      actual_father_match = actual_father_match->next;
      
      
      /* trace */
      if ( gcmd_line.display_info == 130 ) 
	{
	  if (!there_is_new_match)
	    printf("\n?- No");
	  if (actual_father_match)
	    printf("\n...Redo ");
	  there_is_new_match = FALSE;
	}
    }
  
  if ( gcmd_line.display_info == 130 ) {
    if (!there_is_match){
      printf("\n?- No (node)");
    }
  }
  
  return there_is_match;

}

Bool match_state(LTnode *root, State *LState)
{
  int i,j,l;
 Fact *fact;
 int fact_index;
 int *new_bindings = NULL;
 LTmatch *actual_father_match = NULL;
 LTmatch *new_match;
 LTmatch *last_match;
 Bool there_is_match = FALSE;
 Bool there_is_new_match = FALSE;
 Bool tree_head = FALSE;
 LTnode *last_father;
  
  /* searching for previous fathers matches*/
  last_father = get_last_match_father(root);
    
  if (last_father)
    actual_father_match = last_father->matches;
  else
    tree_head = TRUE;
  
  while(actual_father_match || tree_head)
    {
      /*trace*/
      if ( gcmd_line.display_info == 130 ) {
// 	printf("\n **matching state ");
	print_LTnode(root);
	if (actual_father_match)
	  print_partially_bounded_LTnode(root, actual_father_match->bindings);
      }
      
      for ( i = 0; i < LState->num_F; i++ ) 
	{
	  fact_index = LState->F[i];
	  fact =  &(grelevant_facts[fact_index]);
	  
	  /*if ( gcmd_line.display_info == 130 ) {
	    printf("\n>>> Fact[i]:");
	    print_ft_name(fact_index);
	    }*/
	  
	  /* same predicate name */
	  if (fact->predicate == root->pred_index)
	    {
	      
	      if (root->nvars == garity[fact->predicate])
		{
		  
		  new_bindings = (int * ) calloc( LT_num_vars, sizeof( int ) );
		  
		  /* copy father bindings */
		  if (last_father )
		    {
		      copy_bindings(actual_father_match, new_bindings);
		    }
		  
		  /* match bindings */
		  for (j=0; j< root->nvars; j++)
		    {
		      /* free variable */
		      if (root->vars[j].free)
			{
			  
			  new_bindings[root->vars[j].index] = (fact->args)[j];
			  /*  (fact->args)[j] is the index of gconstants */
			  
			}
		      
		      /* bounded variable */
		      else
			{
			  if (new_bindings[root->vars[j].index] != (fact->args)[j])
			    break;
			}
		    }
		  
		  /* there is no match*/
		  if (j != root->nvars)
		    {
		      free_bindings(new_bindings);
		      continue;
		    }
		  else
		    {
		      /* match found */
		      there_is_match = TRUE;
		      there_is_new_match = TRUE;
		      
		      new_match =  (LTmatch * ) calloc( 1, sizeof( LTmatch ) );
		      
		      new_match->bindings = new_bindings;
		      new_match->next = NULL;
		      new_match->index = fact_index;
		      
		      /*trace*/
		      if ( gcmd_line.display_info == 130 ) {
			print_new_variables_values_LTnode(root, new_bindings);
		      }
		      
		      if (root->matches == NULL)
			{
			  root->matches = new_match;
			  last_match = root->matches;
			}
		      else
			{
			  last_match->next = new_match;
			  last_match = last_match->next;
			}
		      /*trace*/
		      /*
			if ( gcmd_line.display_info == 130 ) {
			print_matches(root);
			}
		      */
		    }
		}
	    }
	}
      
      if (tree_head)
	break;
      
      /* compute next binding only if different values for the root variables */
      while (actual_father_match->next)
	{
	  for(l = 0; l < root->nvars; l++)
	    {
	      if (!root->vars[l].free )
		{
		  if (actual_father_match->bindings[root->vars[l].index]!= 
		      actual_father_match->next->bindings[root->vars[l].index])
		    break;
		}
	    }
	  
	  /* Same values for actual_father_match->next*/
	  if (l == root->nvars)
	    {
	      actual_father_match = actual_father_match->next;
	    }
	  else
	    break;
	}
      
      actual_father_match = actual_father_match->next;
      
      /* trace */
      if ( gcmd_line.display_info == 130 ) 
	{
	  if (!there_is_new_match)
	    printf("\n?- No");
	  if (actual_father_match)
	    printf("\n...Redo ");
	  there_is_new_match = FALSE;
	}
    }
  
  if ( gcmd_line.display_info == 130 ) {
    if (!there_is_match){
      printf("\n?- No (node)");
    }
  }
  
  return there_is_match;
}



Bool fact_is_goal(int fact_index){
  int i;
  
  for(i = 0; i < gnum_flogic_goal; i++)
    {
      if (gflogic_goal[i] == fact_index) 
	return TRUE;
    }
  return FALSE;
}


Bool match_rp_adds_and_dels(LTnode *root)
{
  int i,j,l,k;
  Fact *fact;
  int op, eff, lnum_eff;
  int *l_eff;
  int *new_bindings = NULL;
  LTmatch *actual_father_match = NULL;
  LTmatch *new_match;
  LTmatch *last_match;
  Bool there_is_match = FALSE;
  Bool there_is_new_match = FALSE;
  Bool tree_head = FALSE;
  LTnode *last_father;
  
  
  /* searching for previous fathers matches*/
  last_father = get_last_match_father(root);

  if (last_father)
    actual_father_match = last_father->matches;
  else
    tree_head = TRUE;
  
  
  while(actual_father_match || tree_head)
    {
      /*trace*/
      if ( gcmd_line.display_info == 130 ) {
	printf("\n");
	print_LTnode(root);
	if (actual_father_match)
	  print_partially_bounded_LTnode(root, actual_father_match->bindings);
      }

      for(k = 0; k < LT_current_num_RP; k++)
	{
	  op = LT_current_RP[k];
	  
	  if (root->type == PREDICATETYPE_RPADD){
	    lnum_eff = gef_conn[op].num_A;
	    l_eff = gef_conn[op].A;
	  }
	  
	  if (root->type == PREDICATETYPE_RPDEL){
	    lnum_eff = gef_conn[op].num_D;
	    l_eff = gef_conn[op].D;
	  }
	  
	  for (i=0; i < lnum_eff; i++)
	    {
	      eff = l_eff[i];
	      
	      /* do not match RP_ADDs that are goals, since they are in target or achieved*/
 	      if (root->type == PREDICATETYPE_RPADD && fact_is_goal(eff))
 		continue;
	      
	      fact =  &(grelevant_facts[eff]);
	      
	      /* same predicate name */
	      if (fact->predicate == root->pred_index)
		{
		  if (root->nvars == garity[fact->predicate])
		    {
		      new_bindings = (int * ) calloc( LT_num_vars, sizeof( int ) );
		      
		      /* copy father bindings */
		      if (last_father)
			{
			  copy_bindings(actual_father_match, new_bindings);
			}
		      
		      /* match bindings */
		      for (j=0; j< root->nvars; j++)
			{
			  /* free variable */
			  if (root->vars[j].free)
			    {
			      
			      new_bindings[root->vars[j].index] = (fact->args)[j];
			      /*  (fact->args)[j] is the index of gconstants */
			      
			    }
			  
			  /* bounded variable */
			  else
			    {
			      if (new_bindings[root->vars[j].index] != (fact->args)[j])
				break;
			    }
			}
		      
		      /* there is no match*/
		      if (j != root->nvars)
			{
			  free_bindings(new_bindings);
			  continue;
			}
		      else
			{
			  /* match found */
			  there_is_match = TRUE;
			  there_is_new_match = TRUE;
			  
			  new_match =  (LTmatch * ) calloc( 1, sizeof( LTmatch ) );
			  
			  new_match->bindings = new_bindings;
			  new_match->next = NULL;
			  new_match->index = eff;
			  
			  /*trace*/
			  if ( gcmd_line.display_info == 130 ) {
			    print_new_variables_values_LTnode(root, new_bindings);
			  }
			  
			  if (root->matches == NULL)
			    {
			      root->matches = new_match;
			      last_match = root->matches;
			    }
			  else
			    {
			      last_match->next = new_match;
			      last_match = last_match->next;
			    }
			  /*trace*/
			  /*
			    if ( gcmd_line.display_info == 130 ) {
			    print_matches(root);
			    }
			  */
			  
			}
		    }
		}
	    }
	}
      
      if (tree_head)
	break;
      
      /* compute next binding only if different values for the root variables */
      while (actual_father_match->next)
	{
	  for(l = 0; l < root->nvars; l++)
	    {
	      if (!root->vars[l].free )
		{
		  if (actual_father_match->bindings[root->vars[l].index]!= 
		      actual_father_match->next->bindings[root->vars[l].index])
		    break;
		}
	    }
	  
	  /* Same values for actual_father_match->next*/
	  if (l == root->nvars)
	    {
	      actual_father_match = actual_father_match->next;
	    }
	  else
	    break;
	}
      
      actual_father_match = actual_father_match->next;
      
      /* trace */
      if ( gcmd_line.display_info == 130 ) 
	{
	  if (!there_is_new_match)
	    printf("\n?- No");
	  if (actual_father_match)
	    printf("\n...Redo ");
	  there_is_new_match = FALSE;
	}
    }
  
  if ( gcmd_line.display_info == 130 ) {
    if (!there_is_match){
      printf("\n?- No (node)");
    }
  }
  
  return there_is_match;

}


Bool match_landmark(LTnode *root)
{
  int i,j,l;
 Fact *fact;
 int *new_bindings = NULL;
 LTmatch *actual_father_match = NULL;
 LTmatch *new_match;
 LTmatch *last_match;
 Bool there_is_match = FALSE;
 Bool there_is_new_match =FALSE;
 Bool tree_head = FALSE;
 LTnode *last_father;
  
  /* searching for previous fathers matches*/
 last_father = root->father;
 while (last_father)
   {
     if (last_father->matches){
       actual_father_match = last_father->matches;
       /*
       if ( gcmd_line.display_info == 130 ) {
	 printf("\nFather matches");
	 print_matches(last_father);
       }
       */

       break;
     }
     else
       {
	 last_father= last_father->father;
       }

   }

 if (!last_father)
     tree_head = TRUE;


 while(actual_father_match || tree_head)
   {
     /*trace*/
     if ( gcmd_line.display_info == 130 ) {
       printf("\n");
       print_LTnode(root);
       if (actual_father_match)
	 print_partially_bounded_LTnode(root, actual_father_match->bindings);
     }
  
   /*compute landmarks for the current
 state only when necessary: 
     the tree contains the query about a landmark and they have not been computed for the current state */
    if (!landmarks_computed)
       {
	 /*get_landmarks( LT_current_state, FALSE );*/
	 get_landmarks_naive( LT_current_state );
	 landmarks_computed = TRUE;
       }

   for(i = 0; i < gnum_ft_conn; i++)
      {
	if (!gft_conn[i].is_landmark)
	  continue;

	fact =  &(grelevant_facts[i]);

	/* same predicate name */
	if (fact->predicate == root->pred_index)
	  {

	    if (root->nvars == garity[fact->predicate])
	      {
		
		new_bindings = (int * ) calloc( LT_num_vars, sizeof( int ) );
		
		/* copy father bindings */
		if (last_father )
		  {
		    copy_bindings(actual_father_match, new_bindings);
		  }

		/* match bindings */
		for (j=0; j< root->nvars; j++)
		  {
		    /* free variable */
		    if (root->vars[j].free)
		      {
					
			new_bindings[root->vars[j].index] = (fact->args)[j];
			/*  (fact->args)[j] is the index of gconstants */
			
		      }

		    /* bounded variable */
		    else
		      {
			if (new_bindings[root->vars[j].index] != (fact->args)[j])
			  break;
		      }
		  }
		  
		/* there is no match*/
		if (j != root->nvars)
		  {
		    free_bindings(new_bindings);
		    continue;
		  }
		else
		  {
		    /* match found */
		    there_is_match = TRUE;
		    there_is_new_match = TRUE;
		    
		    new_match =  (LTmatch * ) calloc( 1, sizeof( LTmatch ) );
			
		    new_match->bindings = new_bindings;

		    new_match->next = NULL;

		    new_match->index = i;


		    /*trace*/
  		    if ( gcmd_line.display_info == 130 ) {
		      print_new_variables_values_LTnode(root, new_bindings);
		    }

		
		    if (root->matches == NULL)
		      {

			root->matches = new_match;
			last_match = root->matches;

		      }
		    else
		      {
			last_match->next = new_match;
			last_match = last_match->next;
		      }
		    /*trace*/
		    /*
		     if ( gcmd_line.display_info == 130 ) {
		       print_matches(root);
		     }
		    */
		 
		  }
	      }
	  }
      }

   if (tree_head)
     break;
   
   /* compute next binding only if different values for the root variables */
   while (actual_father_match->next)
     {
       for(l = 0; l < root->nvars; l++)
	 {
	   if (!root->vars[l].free )
	     {
	       if (actual_father_match->bindings[root->vars[l].index]!= 
		   actual_father_match->next->bindings[root->vars[l].index])
		 break;
	     }
	 }
       
       /* Same values for actual_father_match->next*/
       if (l == root->nvars)
	 {
	   actual_father_match = actual_father_match->next;
	 }
       else
	 break;
     }
	 
     
   actual_father_match = actual_father_match->next;


   /* trace */
   if ( gcmd_line.display_info == 130 ) 
     {
     if (!there_is_new_match)
       printf("\n?- No");
     if (actual_father_match)
       printf("\n...Redo ");
     there_is_new_match = FALSE;
     }
   }

 if ( gcmd_line.display_info == 130 ) {
   if (!there_is_match){
     printf("\n?- No (node)");
   }
 }

 return there_is_match;
}


Bool match_init(LTnode *root)
{
  int i,j,k,l;
 Fact *fact;
 int *new_bindings = NULL;
 LTmatch *actual_father_match = NULL;
 LTmatch *new_match;
 LTmatch *last_match;
 Bool there_is_match = FALSE;
 Bool there_is_new_match = FALSE;
 Bool tree_head = FALSE;
 LTnode *last_father;

  /* searching for previous fathers matches*/
  last_father = get_last_match_father(root);
    
  if (last_father)
    actual_father_match = last_father->matches;
  else
    tree_head = TRUE;


 while(actual_father_match || tree_head)
   {
      /*trace*/
     if ( gcmd_line.display_info == 130 ) {
       printf("\n");
       print_LTnode(root);
       if (actual_father_match)
	 print_partially_bounded_LTnode(root, actual_father_match->bindings);
     }

   for(i = 1; i < gnum_orig_predicates; i++)
     {
       /* static and initial predicate */
       
       for ( k = 0; k < gnum_initial_predicate[i]; k++ ) 
	 {
	   
	   fact = &(ginitial_predicate[i][k]) ; 
	   
	   /* same predicate name */
	   if (fact->predicate == root->pred_index)
	     {
	       
	       if (root->nvars == garity[fact->predicate])
		 {
		   
		   new_bindings = (int * ) calloc( LT_num_vars, sizeof( int ) );
		   
		   /* copy father bindings */
		   if (last_father )
		     {
		       copy_bindings(actual_father_match, new_bindings);
		     }
		   
		   /* match bindings */
		   for (j=0; j< root->nvars; j++)
		     {
		       /* free variable */
		       if (root->vars[j].free)
			 {
			   
			   new_bindings[root->vars[j].index] = (fact->args)[j];
			   /*  (fact->args)[j] is the index of gconstants */
			   
			 }
		       
		       /* bounded variable */
		       else
			 {
			   if (new_bindings[root->vars[j].index] != (fact->args)[j])
			     break;
			 }
		     }
		   
		   /* there is no match*/
		   if (j != root->nvars)
		     {
		       free_bindings(new_bindings);
		       continue;
		     }
		   else
		     {
		       /* match found */
		       there_is_match = TRUE;
		       there_is_new_match = TRUE;
		       
		       new_match =  (LTmatch * ) calloc( 1, sizeof( LTmatch ) );
		       
		       new_match->bindings = new_bindings;
		       
		       new_match->next = NULL;
		       
		       /*trace*/
		       if ( gcmd_line.display_info == 130 ) {
			 print_new_variables_values_LTnode(root, new_bindings);
		       }
		       
		       
		       if (root->matches == NULL)
			 {
			   root->matches = new_match;
			   last_match = root->matches;
			   
			 }
		       else
			 {
			   last_match->next = new_match;
			   last_match = last_match->next;
			   
			 }
		       
		       
		     }
		 }
	     }
	 }
       
     }
   
   
   if (tree_head)
     break;
   
   /* compute next binding only if different values for the root variables */
   while (actual_father_match->next)
     {
       for(l = 0; l < root->nvars; l++)
	 {
	   if (!root->vars[l].free )
	     {
	       if (actual_father_match->bindings[root->vars[l].index]!= 
		   actual_father_match->next->bindings[root->vars[l].index])
		 break;
	     }
	 }
       
       /* Same values for actual_father_match->next*/
       if (l== root->nvars)
	 {
	   actual_father_match = actual_father_match->next;
	 }
       else
	 break;
     }
	 


   actual_father_match = actual_father_match->next;

   /* trace */
   if ( gcmd_line.display_info == 130 ) 
     {
     if (!there_is_new_match)
       printf("\n?- No");
     if (actual_father_match)
       printf("\n...Redo ");
     there_is_new_match = FALSE;
     }
   }

 if ( gcmd_line.display_info == 130 ) {
   if (!there_is_match){
     printf("\n?- No (node)");
   }
 }

 return there_is_match;
}



// int * dpred_fact_bindings_from_match (LTmatch *current_match, DPmetapred *metapred, Fact *fact){
//   int *new_bindings;
//   int j;
//   
//   if (fact->predicate == metapred->pred_index){
//     if (metapred->nvars == garity[fact->predicate]){
//     
//       new_bindings = (int * ) calloc( LT_num_vars, sizeof( int ) );
//       
//       /* copy father bindings */
//       if (current_match != NULL)
// 	copy_bindings(actual_father_match, new_bindings);
//       
//       
//       /* match bindings */
//       for (j=0; j< metapred->nvars; j++){
// 	/* free variable */
// 	if (metapred->vars[j].free){
// 	  
// 	  new_bindings[metapred->vars[j].index] = (fact->args)[j];
// 	  /*  (fact->args)[j] is the index of gconstants */
// 	}
// 	else{ 	/* bounded variable */
// 	  
// 	  if (new_bindings[metapred->vars[j].index] != (fact->args)[j])
// 	    break;
// 	}
//       }
// 	  
//       /* there is a match*/
//       if (j == metapred->nvars)
// 	return new_bindings;
//     }
//   }
//   /*there is no match*/
//   free_bindings(new_bindings);
//   return NULL;
// } 



int * fact_bindings_from_match (LTmatch *current_match, LTnode *root, Fact *fact){
  int *new_bindings;
  int j;
  
  if (fact->predicate == root->pred_index){
    if (root->nvars == garity[fact->predicate]){
    
      new_bindings = (int * ) calloc( LT_num_vars, sizeof( int ) );
      
      /* copy father bindings */
      if (current_match != NULL)
	copy_bindings(current_match, new_bindings);
      
      
      /* match bindings */
      for (j=0; j< root->nvars; j++){
	/* free variable */
	if (root->vars[j].free){
	  
	  new_bindings[root->vars[j].index] = (fact->args)[j];
	  /*  (fact->args)[j] is the index of gconstants */
	}
	else{ 	/* bounded variable */
	  
	  if (new_bindings[root->vars[j].index] != (fact->args)[j])
	    break;
	}
      }
	  
      /* there is a match*/
      if (j == root->nvars)
	return new_bindings;
    }
  }
  /*there is no match*/
  free_bindings(new_bindings);
  return NULL;
} 
      



Bool match_derived(LTnode *root, State *LState){
  int i,j,l;
  int *new_bindings = NULL;
  LTmatch *actual_father_match = NULL;
  LTmatch *new_match;
  LTmatch *last_match;
  Bool there_is_match = FALSE;
  Bool there_is_new_match = FALSE;
  Bool tree_head = FALSE;
  LTnode *last_father;
  DPclause *dpclause;

  /* searching for previous fathers matches*/
  last_father = get_last_match_father(root);
  
  if (last_father)
    actual_father_match = last_father->matches;
  else
    tree_head = TRUE;
  
  dpclause = &(gDP_bgclauses[root->pred_index]);

  while(actual_father_match || tree_head)
    {
      /*trace*/
      if ( gcmd_line.display_info == 130 ) {
// 	printf("\n **matching state ");
	print_LTnode(root);
	if (actual_father_match)
	  print_partially_bounded_LTnode(root, actual_father_match->bindings);
      }
      
      for ( i = 0; i < dpclause->num_and_literals; i++ ) 
	{
	  if (!dpclause->and_literals[i].dp_state)
	    continue;
// 
// 	  fact_index = LState->F[i];
// 	  fact =  &(grelevant_facts[fact_index]);
// 	  

	  new_bindings = (int * ) calloc( LT_num_vars, sizeof( int ) );
	  
	  /* copy father bindings */
	  if (last_father )
	    {
	      copy_bindings(actual_father_match, new_bindings);
	    }
	  
	  /* match bindings */
	  for (j=0; j< root->nvars; j++)
	    {
	      /* free variable */
	      if (root->vars[j].free)
		{
		  
		  new_bindings[root->vars[j].index] = dpclause->and_literals[i].args[j];
		  /*  (fact->args)[j] is the index of gconstants */
		  
		}
	      
	      /* bounded variable */
	      else
		{
		  if (new_bindings[root->vars[j].index] != dpclause->and_literals[i].args[j])
		    break;
		}
	    }
	  
	  /* there is no match*/
	  if (j != root->nvars)
	    {
	      free_bindings(new_bindings);
	      continue;
	    }
	  else
	    {
	      /* match found */
	      there_is_match = TRUE;
	      there_is_new_match = TRUE;
	      
	      new_match =  (LTmatch * ) calloc( 1, sizeof( LTmatch ) );
	      
	      new_match->bindings = new_bindings;
	      new_match->next = NULL;
	      new_match->index = i;
		      
	      /*trace*/
	      if ( gcmd_line.display_info == 130 ) {
		print_new_variables_values_LTnode(root, new_bindings);
	      }
	      
	      if (root->matches == NULL)
		{
		  root->matches = new_match;
		  last_match = root->matches;
		}
	      else
		{
		  last_match->next = new_match;
		  last_match = last_match->next;
		}
	      /*trace*/
	      /*
		if ( gcmd_line.display_info == 130 ) {
		print_matches(root);
		}
	      */
	    }
	}
      if (tree_head)
	break;
      
      /* compute next binding only if different values for the root variables */
      while (actual_father_match->next)
	{
	  for(l = 0; l < root->nvars; l++)
	    {
	      if (!root->vars[l].free )
		{
		  if (actual_father_match->bindings[root->vars[l].index]!= 
		      actual_father_match->next->bindings[root->vars[l].index])
		    break;
		}
	    }
	  
	  /* Same values for actual_father_match->next*/
	  if (l == root->nvars)
	    {
	      actual_father_match = actual_father_match->next;
	    }
	  else
	    break;
	}
      
      actual_father_match = actual_father_match->next;
      
      /* trace */
      if ( gcmd_line.display_info == 130 ) 
	{
	  if (!there_is_new_match)
	    printf("\n?- No");
	  if (actual_father_match)
	    printf("\n...Redo ");
	  there_is_new_match = FALSE;
	}
    }
  
  if ( gcmd_line.display_info == 130 ) {
    if (!there_is_match){
      printf("\n?- No (node)");
    }
  }
  
  return there_is_match;
}




Bool compute_matches(LTnode *root, State *nextS)
{

  if (root->tag_bounded){
    /*printf(" matching a tag bounded node...");*/
    return match_candidate_bounded_tag(root);
  }
  
  switch(root->type){

  case PREDICATETYPE_CANDIDATE: return match_candidate_ha(root);
  case PREDICATETYPE_TARGET: return match_target_goal(root); 
  case PREDICATETYPE_STATIC: return match_static(root);
  case PREDICATETYPE_EXECUTED: return match_executed(root);
  case PREDICATETYPE_NOTHELPFUL: return match_candidates_ha_and_nonh(root);
  case PREDICATETYPE_RPACTION: return match_rpaction(root);
  case PREDICATETYPE_STATE: return match_state(root, LT_current_state);
  case PREDICATETYPE_ACHIEVEDGOAL: return match_goals_target_and_achieved(root);
  case PREDICATETYPE_RPADD: return match_rp_adds_and_dels(root);
  case PREDICATETYPE_RPDEL: return match_rp_adds_and_dels(root);
  case PREDICATETYPE_NEXT: return match_state(root, nextS);
  case PREDICATETYPE_LANDMARK: return match_landmark(root);
  case PREDICATETYPE_DERIVED: return match_derived(root,nextS);
  case PREDICATETYPE_INIT: return match_init(root);
  case PREDICATETYPE_GOAL:return match_goals_target_and_achieved(root);
  }
  return FALSE;
}


Bool pending_goal(int fact_index)
{
   int i;

   for ( i = 0; i < LT_current_state->num_F; i++ ) {
     if (fact_index == LT_current_state->F[i])
       return FALSE;

   }
   
   return TRUE;
  
}

void fill_predicate_and_action_indexes(LTnode *root)
 {
   int i;
   char name_without_hyphens[MAX_LENGTH];

   if (!root) return;

   if (root->type == NODETYPE_LEAF) return;
   
   if (root->type == PREDICATETYPE_TARGET || root->type == PREDICATETYPE_STATIC ||
       root->type == PREDICATETYPE_ACHIEVEDGOAL || root->type == PREDICATETYPE_STATE ||
       root->type == PREDICATETYPE_RPADD || root->type == PREDICATETYPE_RPDEL ||
       root->type == PREDICATETYPE_LANDMARK || root->type == PREDICATETYPE_NEXT ||
       root->type == PREDICATETYPE_GOAL || root->type == PREDICATETYPE_INIT)
     {
       for (i = 1; i < gnum_orig_predicates; i++)
	 {
	   strcpy(name_without_hyphens,gpredicates[i]);
	   string_replace(name_without_hyphens,"-","_");

	   if (strcasecmp(name_without_hyphens, root->pred) == SAME)
	     {
	       root->pred_index = i;
	       break;
	     }
	 }
     }

   if (root->type ==  PREDICATETYPE_CANDIDATE || root->type == PREDICATETYPE_EXECUTED ||
       root->type == PREDICATETYPE_NOTHELPFUL || root->type == PREDICATETYPE_RPACTION )
     {
       for (i = 0; i < gnum_operators; i++)
	 {

	   strcpy(name_without_hyphens,goperators[i]->name);
	   string_replace(name_without_hyphens,"-","_");

	   if (strcasecmp(name_without_hyphens, root->pred) == SAME)
	     {
	       root->pred_index = i;
	       break;
	     }
	 }
     }

   if (root->type == PREDICATETYPE_DERIVED)
     { 
       for (i = 0; i< gDP_num_clauses; i++){
	 if (strcasecmp(root->pred, gDP_bgclauses[i].head->name)== SAME){
	   root->pred_index = i;
	 }
       }
     }

   
   fill_predicate_and_action_indexes(root->node_yes);
   fill_predicate_and_action_indexes(root->node_no);
 }






void assign_LT_current_state(State *S)
{
  LT_current_state = S;
  landmarks_computed = FALSE;
}

void assign_LT_last_executed_action(int op)
{
  if (op!= -1){
    LT_last_executed_index = op;
    LT_last_executed_action = gop_conn[op].action;
  }
  else
    LT_last_executed_action= NULL;
    
}

void assign_LT_heuristic_info (BfsNode *current)
{
  LT_current_num_RP = current->gnum_RP;
  LT_current_RP = current->g_RP;
}


int action_name_index(char *action_name)
{
  int i;
  char name_without_hyphens[MAX_LENGTH];

  for ( i = 0; i < gnum_operators; i++ ) {

    strcpy(name_without_hyphens,goperators[i]->name);
    string_replace(name_without_hyphens,"-","_");

    if (strcasecmp(action_name,name_without_hyphens) == SAME)
	return i;
  }

  if (strcmp(action_name,"selected") == SAME || strcmp(action_name,"positive") == SAME )
    return 0;
  else if (strcmp(action_name,"rejected") == SAME || strcmp(action_name,"negative") == SAME)
    return 1;

 return -1;
}

void string_replace (char * string, const char *old, const char *new){
  char replaced_string[MAX_LENGTH];
  char* sp;
  int i,j,k;
  
  while((sp=strstr(string, old))!=NULL){
    i=0;
    while(string[i]!=sp[0]){
      replaced_string[i]=string[i];
      i++;
    }
    for(j=0;j<strlen(new);j++){
      replaced_string[i+j]=new[j];
    }

    for(k=0;(i+strlen(old)+k)<strlen(string);k++){
      replaced_string[i+j+k]=string[i+strlen(old)+k];
    }    

    replaced_string[i+j+k]='\0';      
    strcpy(string, replaced_string);  
  }

  return;
}


void string_to_lower(char *string, char *new)
{
  int i;
  for (i =0; i<= strlen(string); i++)
    {
      new[i]=tolower(string[i]);
    }
}

void parse_tilde_line (char *tilde_line){
  string_replace(tilde_line,"|", "");
  string_replace(tilde_line,"?", ""); 
  string_replace(tilde_line,"(", "|");
  string_replace(tilde_line,",", "|");
  string_replace(tilde_line,")", "|");
  string_replace(tilde_line,"]", "|");
  string_replace(tilde_line,"[", "|");
  string_replace(tilde_line,":", "|"); 
  string_replace(tilde_line,"+--yes", "yes");
  string_replace(tilde_line,"+--no", "no");
  string_replace(tilde_line," ", ""); 
  return;
}


/* returns node_type */
/* raquel: seguro que esto se puede hacer mejor*/
int extract_node_type_and_pred(char *tokens, LTnode*node)
{
  
  if (strstr(tokens,"dpand")  || strstr(tokens,"dprec"))
    {
      node->pred = (char*)calloc(strlen(tokens), sizeof(char));   
      strcpy(node->pred, tokens);
      return PREDICATETYPE_DERIVED;
    }
  
  if (strstr(tokens,"nothelpful"))
    {
      node->pred = (char*)calloc(strlen(tokens)-10, sizeof(char));   
      strncpy(node->pred, tokens+11,strlen(tokens)-10);
      return PREDICATETYPE_NOTHELPFUL;
    }

  if (strstr(tokens,"helpful"))
    {
      node->pred = (char*)calloc(strlen(tokens)-7, sizeof(char));   
      strncpy(node->pred, tokens+8,strlen(tokens)-7);
      return PREDICATETYPE_CANDIDATE;
    }

  if (strstr(tokens,"executed"))
    {
      node->pred = (char*)calloc(strlen(tokens)-8, sizeof(char));   
      strncpy(node->pred, tokens+9,strlen(tokens)-8);
      return  PREDICATETYPE_EXECUTED;
    }

  if (strstr(tokens,"rplan"))
    {
      node->pred = (char*)calloc(strlen(tokens)-5, sizeof(char));   
      strncpy(node->pred, tokens+6,strlen(tokens)-5);
      return  PREDICATETYPE_RPACTION;
    }
  
  if (strstr(tokens,"state"))
    {
      node->pred = (char*)calloc(strlen(tokens)-5, sizeof(char));   
      strncpy(node->pred, tokens+6,strlen(tokens)-5);
      return PREDICATETYPE_STATE;
    }
  
  if (strstr(tokens,"achieved"))
    {
      node->pred = (char*)calloc(strlen(tokens)-8, sizeof(char));   
      strncpy(node->pred, tokens+9,strlen(tokens)-8);
      return PREDICATETYPE_ACHIEVEDGOAL;
    }

  if (strstr(tokens,"target_goal"))
    {
      node->pred = (char*)calloc(strlen(tokens)-11, sizeof(char));   
      strncpy(node->pred, tokens+12,strlen(tokens)-11);
      return PREDICATETYPE_TARGET;
    }
  
  if (strstr(tokens,"static_fact"))
    {
      node->pred = (char*)calloc(strlen(tokens)-11, sizeof(char));   
      strncpy(node->pred, tokens+12,strlen(tokens)-11);
      return PREDICATETYPE_STATIC;
    }

  if (strstr(tokens,"rpadd"))
    {
      node->pred = (char*)calloc(strlen(tokens)-5, sizeof(char));   
      strncpy(node->pred, tokens+6,strlen(tokens)-5);
      return PREDICATETYPE_RPADD;
    }

  if (strstr(tokens,"rpdel"))
    {
      node->pred = (char*)calloc(strlen(tokens)-5, sizeof(char));   
      strncpy(node->pred, tokens+6,strlen(tokens)-5);
      return PREDICATETYPE_RPDEL;
    }

  if (strstr(tokens,"landmark"))
    {
      node->pred = (char*)calloc(strlen(tokens)-8, sizeof(char));   
      strncpy(node->pred, tokens+9,strlen(tokens)-8);
      return PREDICATETYPE_LANDMARK;
    }
  
  if (strstr(tokens,"next"))
    {
      node->pred = (char*)calloc(strlen(tokens)-4, sizeof(char));   
      strncpy(node->pred, tokens+5,strlen(tokens)-4);
      return PREDICATETYPE_NEXT;
    }

  if (strstr(tokens,"goal"))
    {
      node->pred = (char*)calloc(strlen(tokens)-4, sizeof(char));   
      strncpy(node->pred, tokens+5,strlen(tokens)-4);
      return PREDICATETYPE_GOAL;
    }

  if (strstr(tokens,"init"))
    {
      node->pred = (char*)calloc(strlen(tokens)-4, sizeof(char));   
      strncpy(node->pred, tokens+5,strlen(tokens)-4);
      return PREDICATETYPE_INIT;
    }



  return -10;  
  
}


int extract_operator_from_action(int ha)
{
  //  int i;

  return gop_conn[ha].action->name_index;
  /*
  for (i = 0; i < gnum_operators; i++)
    {
      if (gop_conn[ha].action->norm_operator->operator == goperators[i])
	return i;
    }
  return -1;
  */
  
}




int compare_proposal_values( const void* a, const void* b ) {
   LTproposal_element* arg1 = ( LTproposal_element*) a;
   LTproposal_element* arg2 = ( LTproposal_element*) b;

   if (arg1->value < arg2->value) return 1;
   else return -1;
 }    


LTnode * load_tilde_tag (char * tilde_line, Bool is_bindings_tree){

  char *tokens;
  char old_tilde_line[MAX_LENGTH];
  LTnode* node;

  int nvars;
  char var_names[MAX_VARS][MAX_LENGTH];
  int i,j;  

  strcpy(old_tilde_line,tilde_line);
  tokens=strtok(tilde_line,"|");

  if (is_bindings_tree && (strcmp(tokens,"selected")==SAME || strcmp(tokens,"rejected")==SAME)){
    /* no tag. The tree is directly a proposal node*/
    

    return load_tilde_node(old_tilde_line,NULL,TRUE, TRUE);
  }


  node = (LTnode *) calloc(1,sizeof(LTnode));
  node->node_yes=NULL;
  node->node_no=NULL;
  node->father = NULL;

  node->type = NODETYPE_TAG;

  /*selected*/
  node->pred = (char*)calloc(strlen(tokens)+1, sizeof(char));
  strcpy(node->pred,tokens);
  

  /* Buffering the variables */
  nvars=0;
  while((tokens=strtok(NULL,"|"))!=NULL){              
      if(strcmp(tokens,"\n")!=0){

	/* skipping the example and problem variables*/	
	if(strcmp("-A",tokens) == SAME || 
	   (domain_with_static_facts && strcmp("-B",tokens)== SAME))
	   continue;
	strcpy(var_names[nvars],tokens); 
	nvars ++;
	         
      }
  }
  
  /* Removing the last variable */  
    nvars --;


    /* Filling the variables of the node */  
    node->nvars=nvars;
    node->vars = (LTnodevar * )calloc( node->nvars, sizeof( LTnodevar ));
    for(i=0;i<node->nvars;i++){ 

      if(strchr(var_names[i],'-')!=NULL){/* new variable */
	    node->vars[i].free=TRUE;
 
	    /* search the variable in the global table of variable. Since the
	       table is common to all trees a free variable in a tree can be
	       in the global table*/
	    for(j=0;j<LT_num_vars;j++){
	      if (strcmp(LT_var_table[j].name, &var_names[i][1])==0){
		break;
	      }
	    }
	    
	    if (j == LT_num_vars) { /*varible not in the global table*/
		LT_var_table[LT_num_vars].name= (char*)calloc(strlen(var_names[i])+1, sizeof(char));
		strcpy(LT_var_table[LT_num_vars].name, &var_names[i][1]);
		LT_var_table[LT_num_vars].index=LT_num_vars;
		node->vars[i].index=LT_num_vars;
		LT_num_vars++;
	      }
	      else { /*variable in the global table*/
		node->vars[i].index=j;	   
	      }
      }
      else { /* not free variable */
	node->vars[i].free=FALSE; 	
	/*necesarily the variable should be in the global table*/
        for(j=0;j<LT_num_vars;j++){
	      if (strcmp(LT_var_table[j].name, var_names[i])==0){
		break;
	      }
	}
	node->vars[i].index=j;	   
      }
    }



    return node;
    
}


LTnode * load_tilde_node (char * tilde_line, LTnode * father, Bool is_proposal_tree, Bool with_example_vars){
  
  LTnode *node;
  char * tokens;
  int proposal_index, op_index;

//   printf("\n\n%s", tilde_line);

  /* Creating the node */
  node = (LTnode *) calloc(1,sizeof(LTnode));
  node->node_yes=NULL;
  node->node_no=NULL;
  node->father=father;
  node->tag_bounded = FALSE;

  /* Linking the node to the father */
  if(node->father!=NULL){
    if(node->father->node_yes==NULL){
      node->father->node_yes=node;
    }else{      
      node->father->node_no=node;
    }
  }
 
  /* Leaf Node */
  if(strstr(tilde_line,"||")!=NULL || is_proposal_tree){             
    int nproposals;
    char proposals_names[MAX_OPERATORS][MAX_LENGTH];
    float proposals_values[MAX_OPERATORS];
    int i;
    
    node->type = NODETYPE_LEAF;
    if (!is_proposal_tree){
      tokens=strtok(tilde_line,"||"); 
      tokens=strtok(NULL,"|");
    }
    else
      tokens=strtok(tilde_line,"|");        

    node->proposal= (LTproposal * ) calloc(1, sizeof(LTproposal ) );
    node->proposal->class = (char *) calloc(strlen(tokens) + 1,sizeof(char));
    strcpy(node->proposal->class, tokens);

    tokens=strtok(NULL,"|");
    
    /* Buffering the leaf values */    
    nproposals=0;
    while((tokens=strtok(NULL,"|"))!=NULL){
      if(strcmp(tokens,"\n")!=0){	
	strcpy(proposals_names[nproposals],tokens);
	tokens=strtok(NULL,"|");	
	proposals_values[nproposals]=atof(tokens);
	nproposals++;
      }
    }
    
    node->proposal->nactions=nproposals;
    /* Filling the leaf values of the node */  
    op_index = -1; /*I will use this as counter if it doesn't refer to an action name*/
    for(i=0;i<node->proposal->nactions;i++){
      proposal_index = action_name_index(proposals_names[i]);
      if (proposal_index >= 0)
	op_index = proposal_index;
      else
	op_index++; 

      (node->proposal->elements[op_index]).action_type=(char*)calloc(strlen(proposals_names[i])+1, sizeof(char));
      strcpy((node->proposal->elements[op_index]).action_type,proposals_names[i]);
      (node->proposal->elements[op_index]).value = proposals_values[i];
      /*      (node->proposal->elements[i]).op_index= action_name_index(proposals_names[i]);*/

    }
    /* order the proposal in decreasing values */
    /*    qsort(&(node->proposal->elements), node->proposal->nactions,
	  sizeof(LTproposal_element), compare_proposal_values)*/
  
   
  }else{  
  /* Test Node */     
    int nvars;
    char var_names[MAX_VARS][MAX_LENGTH];
    int i,j;  
    //node->type = NODETYPE_TEST; 
        
    tokens=strtok(tilde_line,"|");
    if((strcmp(tokens,"yes")!=0)&&(strcmp(tokens,"no")!=0)){ /*root node*/
      node->type = extract_node_type_and_pred(tokens,node);
      /*
      node->pred = (char*)calloc(strlen(tokens)+1, sizeof(char));
      strcpy(node->pred,tokens);
      */
    }else{
      tokens=strtok(NULL,"|");
      node->type = extract_node_type_and_pred(tokens,node);
      /*
      node->pred = (char*)calloc(strlen(tokens)+1, sizeof(char));      
      strcpy(node->pred,tokens);      
      */
    }    
    
    /* Buffering the variables */
    nvars=0;
    while((tokens=strtok(NULL,"|"))!=NULL){              
      if(strcmp(tokens,"\n")!=0){
	/* skipping the example and problem variables*/	
	
	if (with_example_vars){
	  if(strcmp("A",tokens)==SAME ||  (domain_with_static_facts && strcmp("B",tokens)==SAME))
	    continue;
	}
	strcpy(var_names[nvars],tokens); 
	nvars ++;
      }
    }     

    /* Filling the variables of the node */  
    node->nvars=nvars;
    node->vars = (LTnodevar * )calloc( node->nvars, sizeof( LTnodevar ));
    for(i=0;i<node->nvars;i++){ 

      if(strchr(var_names[i],'-')!=NULL){/* new variable */
	    node->vars[i].free=TRUE;
 
	    /* search the variable in the global table of variable. Since the
	       table is common to all trees a free variable in a tree can be
	       in the global table*/
	    for(j=0;j<LT_num_vars;j++){
	      if (strcmp(LT_var_table[j].name, &var_names[i][1])==0){
		break;
	      }
	    }
	    
	    if (j == LT_num_vars) { /*varible not in the global table*/
		LT_var_table[LT_num_vars].name= (char*)calloc(strlen(var_names[i])+1, sizeof(char));
		strcpy(LT_var_table[LT_num_vars].name, &var_names[i][1]);
		LT_var_table[LT_num_vars].index=LT_num_vars;
		node->vars[i].index=LT_num_vars;
		LT_num_vars++;
	      }
	      else { /*variable in the global table*/
		node->vars[i].index=j;	   
	      }
      }
      else { /* not free variable */
	node->vars[i].free=FALSE; 	
	/*necesarily the variable should be in the global table*/
        for(j=0;j<LT_num_vars;j++){
	      if (strcmp(LT_var_table[j].name, var_names[i])==0){
		break;
	      }
	}
	node->vars[i].index=j;	   
      }
    }


	    
	/*
	  if(strchr(var_names[i],'-')!=NULL){*//* new variable */	
	/*
	LT_var_table[LT_num_vars].name= (char*)calloc(strlen(var_names[i])+1, sizeof(char));
	strcpy(LT_var_table[LT_num_vars].name, &var_names[i][1]);
	LT_var_table[LT_num_vars].index=LT_num_vars;
	node->vars[i].index=LT_num_vars;
	node->vars[i].free=TRUE; 
	LT_num_vars++;
	}else{*/ /* old variable */	
	/*
	for(j=0;j<LT_num_vars;j++){
	  if (strcmp(LT_var_table[j].name,var_names[i])==0){
	    node->vars[i].index=j;	   
	    node->vars[i].free=FALSE; 
	  }
	}
	}   */               
  }

  return node;      
}

/* */
/* Reading a TILDE output file */
/* */

LTnode * load_tilde_tree(const char *tilde_out_file_name, Bool is_bindings_tree){
  char tmp[MAX_LENGTH] = "";  
  char line[MAX_LENGTH] = "";
  int line_counter =0;
  Bool intree;    
  LTnode *tree;
  int j;

  LTnode * frontier_node=NULL;

  FILE *fp;
    
  if((fp = fopen(tilde_out_file_name, "r")) == NULL){
    sprintf(tmp, "\nROLLER: can't find tree file: %s\n\n", tilde_out_file_name);
    perror(tmp);
    exit(-1);
  }
  
  /* Adding the Example Variable */
  /*  LT_num_vars=0;  */

  for(j=0;j<LT_num_vars;j++){
    if (strcmp(LT_var_table[j].name,"A")==0){
      break;
    }
  }

  if (j == LT_num_vars) { /*new variable */
    LT_var_table[LT_num_vars].name= (char*)calloc(strlen("A")+1, sizeof(char));
    strcpy(LT_var_table[LT_num_vars].name, "A");
    LT_var_table[LT_num_vars].index=LT_num_vars;
    LT_num_vars++;
  }
  
  /* Adding the Problem Variable */
  for(j=0;j<LT_num_vars;j++){
    if (strcmp(LT_var_table[j].name,"B")==0){
      break;
    }
  }
  
  if (j == LT_num_vars) { /*new variable */
    LT_var_table[LT_num_vars].name= (char*)calloc(strlen("B")+1, sizeof(char));
    strcpy(LT_var_table[LT_num_vars].name, "B");
    LT_var_table[LT_num_vars].index=LT_num_vars;
    LT_num_vars++;
  }
  
  intree=FALSE;
  line_counter=0;
  while(fgets(line,sizeof(line), fp)){    
    if(strstr(line, "Equivalent prolog program with probability distributions:")!=NULL) intree=FALSE;    
    
    if((intree==TRUE)&&(strcmp(line,"\n")!=0)){                
      parse_tilde_line(line);
      /* Adding the tree Tag*/
      if(line_counter==0)
	{
	  tree = load_tilde_tag(line,is_bindings_tree);
	}
      /* Adding the root node of the tree */
      if(line_counter==1){
	
	tree->node_yes = load_tilde_node(line,tree,FALSE, TRUE);
	frontier_node = tree->node_yes;
      }
      /* Adding rest of the nodes of the tree */
      if(line_counter>1){
	if(strstr(line,"||")==NULL){ /* Test Node */	  
	  frontier_node=load_tilde_node(line, frontier_node, FALSE, TRUE);
	}else{ /* Leaf Node */
	  if(strstr(line,"yes")!=NULL){ /* Yes Node */	    
	    load_tilde_node(line, frontier_node, FALSE, TRUE);
	  }
	  if(strstr(line,"no")!=NULL){ /* No Node */	    
	    load_tilde_node(line, frontier_node,FALSE, TRUE);
	    while((frontier_node!=NULL)&&(frontier_node->node_no!=NULL)) frontier_node=frontier_node->father;	    
	  }
	}
      }
      line_counter++;
    }
    
    if(strstr(line, "Compact notation of tree:")!=NULL) intree=TRUE;
  }  
  
  fclose(fp);
  return tree;
}


/* */
/* Reading a Derived predicate Tree learned with TILDE
   This is different from ROLLER because we don't have example vars */
/* */

LTnode * load_tilde_derived_pred_tree(const char *tilde_out_file_name){
  char tmp[MAX_LENGTH] = "";  
  char line[MAX_LENGTH] = "";
  int line_counter =0;
  Bool intree;    
  LTnode *tree;

  LTnode * frontier_node=NULL;

  FILE *fp;
    
  if((fp = fopen(tilde_out_file_name, "r")) == NULL){
    sprintf(tmp, "\nDPTREES: can't find tree file: %s\n\n", tilde_out_file_name);
    perror(tmp);
    exit(-1);
  }

  intree=FALSE;
  line_counter=0;
  while(fgets(line,sizeof(line), fp)){    
    if(strstr(line, "Equivalent prolog program with probability distributions:")!=NULL) intree=FALSE;    
    
    if((intree==TRUE)&&(strcmp(line,"\n")!=0)){                
      parse_tilde_line(line);

      if(line_counter==0)
	{
	  tree = load_tilde_node(line,NULL,FALSE, FALSE);
	  frontier_node = tree;
	}

      /* Adding rest of the nodes of the tree */
      if(line_counter>0){
	if(strstr(line,"||")==NULL){ /* Test Node */	  
	  frontier_node=load_tilde_node(line, frontier_node, FALSE, FALSE);
	}else{ /* Leaf Node */
	  if(strstr(line,"yes")!=NULL){ /* Yes Node */	    
	    load_tilde_node(line, frontier_node, FALSE, FALSE);
	  }
	  if(strstr(line,"no")!=NULL){ /* No Node */	    
	    load_tilde_node(line, frontier_node,FALSE, FALSE);
	    while((frontier_node!=NULL)&&(frontier_node->node_no!=NULL)) frontier_node=frontier_node->father;	    
	  }
	}
      }
      line_counter++;
    }
    
    if(strstr(line, "Compact notation of tree:")!=NULL) intree=TRUE;
  }  
  
  fclose(fp);
  return tree;
}



void load_all_tilde_trees(void)
{
  char operators_tree_file[MAX_LENGTH] = "";
  
  char bindings_tree_file[MAX_LENGTH] = "";

  char operator_name_lower[MAX_LENGTH] = "";

  int i;

  check_static_facts();
  fill_general_action_indexes();

  LT_num_vars = 0;
  
  sprintf(operators_tree_file,"%s%s",gcmd_line.knowledge_path, "siblings-episodes/tilde/siblings-episodes.out");

  LT_tree_head =  load_tilde_tree(operators_tree_file,FALSE);

  fill_predicate_and_action_indexes(LT_tree_head);

  printf("\n\nROLLER: tree loaded %s",operators_tree_file);
  fflush(stdout);


  for (i = 0; i < gnum_operators; i++)
    {
      string_to_lower(goperators[i]->name, operator_name_lower);

     sprintf(bindings_tree_file,"%s%s-episodes/tilde/%s-episodes.out",gcmd_line.knowledge_path, operator_name_lower, operator_name_lower);

     LT_tree_bindings_head[i] =  load_tilde_tree(bindings_tree_file,TRUE);
     fill_predicate_and_action_indexes(LT_tree_bindings_head[i]);
     set_tagbounded(LT_tree_bindings_head[i]);

     printf("\n\nROLLER: tree loaded %s",bindings_tree_file);
     fflush(stdout);      
   }
  n_LTtree_bindings = gnum_operators;
 
  /* trace for printing all loaded trees*/
  
  if ( gcmd_line.display_info == 130 ) {
  printf("\n");
  print_LTtree(LT_tree_head,0);

  for (i = 0; i < gnum_operators; i++)
    {
      printf("\n\nBindings tree for operator: %s \n", goperators[i]->name );
      print_LTtree(LT_tree_bindings_head[i],0);
   }
  printf("\n\n");
  }  

}
  


void load_tilde_dp_trees(void)
{
  char dptree_file[MAX_LENGTH] = "";
    
  LT_num_vars = 0;

  sprintf(dptree_file,"%s%s",gcmd_line.knowledge_path, "tilde/planning-episodes.out");

  LT_tree_head =  load_tilde_derived_pred_tree(dptree_file);

  fill_predicate_and_action_indexes(LT_tree_head);

  printf("\n\nDPTREES: tree loaded %s",dptree_file);
  fflush(stdout);

  /* trace for printing all loaded trees*/
  
  if ( gcmd_line.display_info == 130 ) {
  printf("\n");
  print_LTtree(LT_tree_head,0);
  }  
}





void print_LTtype(int type){
  switch(type){
  case PREDICATETYPE_CANDIDATE: fprintf(stdout,"helpful_");
    break;
  case PREDICATETYPE_TARGET: fprintf(stdout,"target_goal_");
    break;
  case PREDICATETYPE_STATIC: fprintf(stdout,"static_fact_");
    break;
  case PREDICATETYPE_EXECUTED: fprintf(stdout,"executed_");  
    break;
  case PREDICATETYPE_NOTHELPFUL: fprintf(stdout, "nothelpful_");
    break;
  case PREDICATETYPE_RPACTION: fprintf(stdout, "rpaction_");
    break;
  case PREDICATETYPE_STATE: fprintf(stdout, "state_");
    break;
  case PREDICATETYPE_ACHIEVEDGOAL: fprintf(stdout, "achieved_");
    break;
  case PREDICATETYPE_RPADD: fprintf(stdout, "rpadd_");
    break;
  case PREDICATETYPE_RPDEL: fprintf(stdout, "rpdel_");
    break;
  case PREDICATETYPE_LANDMARK: fprintf(stdout, "landmark_");
    break;
  case PREDICATETYPE_NEXT: fprintf(stdout, "next_");
    break;
  case PREDICATETYPE_GOAL: fprintf(stdout, "goal_");
    break;
  case PREDICATETYPE_INIT: fprintf(stdout, "init_");
    break;
  case -1: fprintf(stdout," ");
    break;    
  }
  
  return;    
}



void print_LTproposal(LTproposal *prop)
{
  int i;
  
  for (i = 0; i < prop->nactions; i++)
    {

      fprintf(stdout,"%s",(prop->elements[i]).action_type);
      fprintf(stdout, " ");
      fprintf(stdout,"%f", (prop->elements[i]).value);
      fprintf(stdout, " ");

    }

  return;
}


void print_LTspace(int tab)
{
  int i;


  for (i = 0; i < tab; i++)
    {
      fprintf(stdout," ");
    }

  return;
}

void print_LTnode(LTnode *node)  
{
  int i;

  printf("\n");
 if (node->type == NODETYPE_LEAF){
    print_LTproposal(node->proposal);
  }else{    
    print_LTtype(node->type);
    fprintf(stdout,"%s", node->pred);
    fprintf(stdout,"(");
    for(i = 0; i < node->nvars; i++)
      {

	if (node->vars[i].free)
	  fprintf(stdout,"-");

	if (i< node->nvars-1)
	  fprintf(stdout,"?%s ", LT_var_table[node->vars[i].index].name);
	else
	  fprintf(stdout,"?%s", LT_var_table[node->vars[i].index].name); 
      }
    
    fprintf(stdout,")");

 }
}

void print_partially_bounded_LTnode(LTnode *node, int *bindings)
{
  int i;

  printf("\n?- ");
 if (node->type == NODETYPE_LEAF){
    print_LTproposal(node->proposal);
  }else{    
    print_LTtype(node->type);
    fprintf(stdout,"%s", node->pred);
    fprintf(stdout,"(");
    for(i = 0; i < node->nvars; i++)
      {

	if (node->vars[i].free)
	  {
	    fprintf(stdout,"-");
	    

	    if (i< node->nvars-1)
	      fprintf(stdout,"?%s ", LT_var_table[node->vars[i].index].name);
	    else
	      fprintf(stdout,"?%s", LT_var_table[node->vars[i].index].name); 
	  }
	else
	  {
	    if (i< node->nvars-1)
	      fprintf(stdout,"%s ", gconstants[bindings[node->vars[i].index]]);
	    else
	      fprintf(stdout,"%s",  gconstants[bindings[node->vars[i].index]]); 

	  }
      }
    
    fprintf(stdout,")?");

 }
}

void print_new_variables_values_LTnode(LTnode *node, int *bindings)
{
  int i;
  Bool with_free= FALSE;

  printf("\n");
 if (node->type == NODETYPE_LEAF){
    print_LTproposal(node->proposal);
  }else{    
    for(i = 0; i < node->nvars; i++)
      {
	if (node->vars[i].free)
	  {
	    with_free = TRUE;
	    fprintf(stdout,"?- ?%s = %s\n", LT_var_table[node->vars[i].index].name,
		    gconstants[bindings[node->vars[i].index]]);;

	  }
      }
    
    if (!with_free)
      printf("?- Yes");

 }
}

void print_LTtree(LTnode *root,int tab){
  int i; 

  if(root==NULL) return;
  
  
  if (root->type == NODETYPE_LEAF){
    print_LTproposal(root->proposal);
  }else{    
    print_LTtype(root->type);
    fprintf(stdout,"%s", root->pred);
    fprintf(stdout,"(");
    for(i = 0; i < root->nvars; i++)
      {

	if (root->vars[i].free)
	  fprintf(stdout,"-");

	if (i< root->nvars-1)
	  fprintf(stdout,"?%s ", LT_var_table[root->vars[i].index].name);
	else
	  fprintf(stdout,"?%s", LT_var_table[root->vars[i].index].name); 
      }
    
    fprintf(stdout,")?");
    
    fprintf(stdout,"\n");
    print_LTspace(tab+3);      
    fprintf(stdout,"yes:");
    print_LTtree(root->node_yes, tab+3);
    
    fprintf(stdout,"\n");
    print_LTspace(tab+3);
    fprintf(stdout,"no :");
    print_LTtree(root->node_no,tab+3);    
  }  
  
  fflush(stdout);
  return;
}

void check_static_facts(void)  
{
  int i;

  //temporally disabling this, all domains will be as they have statics
//    domain_with_static_facts = FALSE;
  domain_with_static_facts = TRUE;
  return;

  /* predicate 0 is the in-build predicate: "=" */
  for(i = 1; i < gnum_orig_predicates; i++)
    {
      if (!gis_added[i] && !gis_deleted[i]){
	  /* static predicate */
	domain_with_static_facts = TRUE;
	return;
      }

    }
  
  if (gStatic_literals){
    domain_with_static_facts = TRUE;
  }
    
  
}




void fill_general_action_indexes(void)
{
  int i,j;
  Action *action;


  for(i = 0; i < gnum_op_conn; i++)
    {
      action = gop_conn[i].action;
      
      for (j =0; j < gnum_operators; j++)
	{
	  if (strcasecmp(action->name, goperators[j]->name) == SAME)
	      action->name_index=j;
	}
    }
}



void print_context (void)
{
  int i, goal,k,j,eff,op;
  Fact *fact;

   printf("\n\n**CONTEXT**");
   printf("\nSTATE");
   print_State(*LT_current_state);
   printf("\n");

  /*candidates*/
   printf("\nHELPFULS\n");
   for(i = 0; i < gnum_H; i++)
      {
        print_op_name(gH[i]);
	printf("\n");
      }

  /*static facts*/
   printf("\nSTATIC FACTS\n");


   for(i = 1; i < gnum_orig_predicates; i++)
      {
	if (!gis_added[i] && !gis_deleted[i])
	  /* static predicate */
	  {
	   for ( k = 0; k < gnum_initial_predicate[i]; k++ ) 
	     {
	     fact = &(ginitial_predicate[i][k]) ; 
	     print_Fact(fact);
	     printf("\n");
	     }
	  }
      }


  /*target goals*/
   printf("\nTARGET GOALS\n");

   for(i = 0; i < gnum_flogic_goal; i++)
      {
	goal = gflogic_goal[i]; /* fact index */
	
	/* goal in current state, not pending, then continue */
	if (pending_goal(goal))
	  {
	    fact =  &(grelevant_facts[goal]);
	    print_Fact(fact);
	    printf("\n");

	  }
      }

  /*executed*/
   printf("\nEXECUTED\n");
   if (LT_last_executed_action)
     {
       print_Action_name(LT_last_executed_action);
       printf("\n");
     }

   
   /*rp actions*/
   printf("\nNumRP %d Current numRP %d", gnum_RP, LT_current_num_RP);
   printf("\nRELAXED PLAN ACITONS\n");
   for(i = 0; i < LT_current_num_RP; i++){
     op = LT_current_RP[i];
     print_op_name(op);
     printf("\n");
   }


   /* rp adds */
   printf("\n RP ADDED\n");
   for(i = 0; i < LT_current_num_RP; i++){
     op = LT_current_RP[i];
     for (j=0; j < gef_conn[op].num_A; j++){
       eff = gef_conn[op].A[j];
       fact =  &(grelevant_facts[eff]);
       print_Fact(fact);
       printf("\n");
     }
   }
   
   /* rp dels */
   printf("\n RP DELETED\n");
   for(i = 0; i < LT_current_num_RP; i++){
     op = LT_current_RP[i];
     for (j=0; j < gef_conn[op].num_D; j++){
       eff = gef_conn[op].D[j];
       fact =  &(grelevant_facts[eff]);
       print_Fact(fact);
       printf("\n");
     }
   }
   
   
  
   
}

void print_matches(LTnode *node)
{
  LTmatch *actual_match= node->matches;
  int i;

  while(actual_match)
    {
      printf("\n");
      printf("-->match:");
      for (i =0; i< node->nvars; i++)
	{
	  printf("%s=%s ",LT_var_table[node->vars[i].index].name, 
		 gconstants[actual_match->bindings[node->vars[i].index]]);
	  fflush(stdout);
	}

      actual_match = actual_match->next;
    }
printf("\n");
}

void pruebaLT(void)
{
  /*  
 LT_num_vars = 0;

 LTnode *node1 = (LTnode * ) calloc( 1, sizeof( LTnode ) );
 node1->type = 0;
 node1->pred = "unstack";
 node1->nvars = 2;
 node1->vars = (LTnodevar * ) calloc( node1->nvars, sizeof( LTnodevar ) );


 LTvar varC;
 varC.name="C";
 varC.index=0;
 // varC.value=-1;
 LT_var_table[0]= varC;
 LT_num_vars++;


 LTvar varD;
 varD.name="D";
 varD.index=1;
 // varD.value=-1;
 LT_var_table[1]= varD;
 LT_num_vars++;

 node1->vars[0].index=0;
 node1->vars[0].free=TRUE;
 node1->vars[1].index=1;
 node1->vars[1].free=TRUE;

 node1->father = NULL;


 LTnode *node2 = (LTnode * ) calloc( 1, sizeof( LTnode ) );
 node2->type = 1;
 node2->pred = "on";
 node2->nvars = 2;
 node2->vars = (LTnodevar * ) calloc( node1->nvars, sizeof( LTnodevar ) );

 LTvar varE;
 varE.name="E";
 varE.index=2;
 // varE.value=-1;
 LT_var_table[2]= varE;
 LT_num_vars++;

 node2->vars[0].index=0;
 node2->vars[0].free=FALSE;
 node2->vars[1].index=2;
 node2->vars[1].free=TRUE;
 
 node1->node_yes=node2;
 node2->father =node1;




 LTnode *node3 = (LTnode * ) calloc( 1, sizeof( LTnode ) );
 node3->type = 0;
 node3->pred = "unstack";
 node3->nvars = 2;
 node3->vars = (LTnodevar * ) calloc( node1->nvars, sizeof( LTnodevar ) );
 

 LTvar varH;
 varH.name="H";
 varH.index=3;
 // varH.value=-1;
 LT_var_table[3]= varH;
 LT_num_vars++;



 LTvar varI;
 varI.name="I";
 varI.index=4;
 LT_var_table[4]= varI;
 LT_num_vars++;


 node3->vars[0].index=0;
 node3->vars[0].free=FALSE;
 node3->vars[1].index=3;
 node3->vars[1].free=TRUE;


// node3->vars[1].index=4;
// node3->vars[1].free=TRUE;


 node1->node_no=node3;
 node3->father =node1;



 LTnode *node4 = (LTnode * ) calloc( 1, sizeof( LTnode ) );
 node4->type = -1;

 LTproposal *Prop1 = (LTproposal * ) calloc(1, sizeof(LTproposal ) );

 Prop1->nactions = 2;

 Prop1->action_type = (char **) calloc(2, sizeof(char * ) );
 Prop1->value =  (float *) calloc(2, sizeof(float) );

 Prop1->action_type[0]="put_down";
 Prop1->value[0]= 6.0;


 Prop1->action_type[1]="pick_up";
 Prop1->value[1]= 0.0;

 node4->proposal=Prop1;
 node4->father = node2;
 node2->node_yes=node4;


 
 LTnode *node5 = (LTnode * ) calloc( 1, sizeof( LTnode ) );
 node5->type = -1;

 LTproposal *Prop2 = (LTproposal * ) calloc(1, sizeof(LTproposal ) );

 Prop2->nactions = 2;

 Prop2->action_type = (char **) calloc(2, sizeof(char * ) );
 Prop2->value =  (float *) calloc(2, sizeof(float) );

 Prop2->action_type[0]="put_down";
 Prop2->value[0]= 0.0;


 Prop2->action_type[1]="pick_up";
 Prop2->value[1]= 8.0;

 node5->proposal=Prop2;
 node5->father = node2;
 node2->node_no=node5;

 
 LTnode *node6 = (LTnode * ) calloc( 1, sizeof( LTnode ) );
 node6->type = -1;

 LTproposal *Prop3 = (LTproposal * ) calloc(1, sizeof(LTproposal ) );

 Prop3->nactions = 2;

 Prop3->action_type = (char **) calloc(2, sizeof(char * ) );
 Prop3->value =  (float *) calloc(2, sizeof(float) );

 Prop3->action_type[0]="put_down";
 Prop3->value[0]= 5.0;


 Prop3->action_type[1]="pick_up";
 Prop3->value[1]= 5.0;

 node6->proposal=Prop3;
 node6->father = node3;
 node3->node_yes = node6;


 
 LTnode *node7 = (LTnode * ) calloc( 1, sizeof( LTnode ) );
 node7->type = -1;

 LTproposal *Prop4 = (LTproposal * ) calloc(1, sizeof(LTproposal ) );

 Prop4->nactions = 2;

 Prop4->action_type = (char **) calloc(2, sizeof(char * ) );
 Prop4->value =  (float *) calloc(2, sizeof(float) );

 Prop4->action_type[0]="put_down";
 Prop4->value[0]= 10.0;


 Prop4->action_type[1]="pick_up";
 Prop4->value[1]= 0.2;

 node7->proposal=Prop4;
 node7->father = node3;
 node3->node_no = node7;
 
 LT_tree_head=node1;

 printf("\n");
 print_LTtree(LT_tree_head,0);
 
 // initialize_matches(LT_tree_head);

 //fill_predicate_indexes(LT_tree_head);
 */

  char filename[MAX_LENGTH] = "/home/rfuentet/planificadores/domains/satellite/roller/siblings-episodes/tilde/siblings-episodes.out\0";

  LTnode * tree;
  tree=load_tilde_tree(filename, FALSE); 
       
  print_LTtree(tree,0);  
  printf("\n"); 
}
