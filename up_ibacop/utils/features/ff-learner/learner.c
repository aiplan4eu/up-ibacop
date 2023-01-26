/*********************************************************************
 *
 * File: learner.c
 *
 * Description: Metric-ff Extension for handling the generation of 
 * training examples for different planning-learners
 *
 * Author: Tomas de la Rosa 2009
 *
 * 
 *********************************************************************/ 

#include "ff.h"
#include "learner.h"
#include "memory.h"
#include "output.h"
#include "relax.h"

char prolog_string[MAX_LENGTH];
char temp_string[MAX_LENGTH];
int gdraw_line;
int gdraw_node_num;

long int record_node_counter;


char * toStr_prolog( char *string)
{
  int i; 
  
  for (i=0; string[i] !='\0'; i++){
    if (string[i] == '-') 
      prolog_string[i] = '_';
    else
      prolog_string[i] = tolower(string[i]);
  }
  prolog_string[i] = '\0';
  return prolog_string;
}

char * toStr_lower( char *string)
{
  int i; 
  
  for (i=0; string[i] !='\0'; i++){
    temp_string[i] = tolower(string[i]);
  }
  temp_string[i] = '\0';
  return temp_string;
}




Bool is_target_goal(int fact_index, State *S)
{
   int i;

   for ( i = 0; i < S->num_F; i++ ) {
     if (fact_index == S->F[i])
       return FALSE;
   }
   
   return TRUE;
}



SolPath *new_SolPath(BfsNode *inode){
  SolPath *result = (SolPath *) calloc(1, sizeof(SolPath));
  result->node = inode;
  result->next = NULL;
  
  return result;
}


Solution *new_Solution(BfsNode *last, Bool tag_solution){
  Solution *isol = (Solution *) calloc(1, sizeof(Solution));
  SolPath *isol_path, *ahead;
  BfsNode *inode;
  int i;
 
  //printf("\n Solution trace");
  ahead = NULL;
  isol->last_node = last;
  for (inode = last, i=1; inode->op != -1; inode = inode->father, i++){
    //   printf("\n NumEff %d", gop_conn[inode->op].num_E);
    isol_path = new_SolPath(inode);
    isol_path->next = ahead;
    ahead = isol_path;
    if (tag_solution)
      inode->tag = 1;  // in solution path
  }
  
  isol->path = isol_path;
  isol->path_len = i-1;
  isol->paralelism = 1.0;
  isol->difficulty = 1.0; 
  isol->next = NULL; 

  if (tag_solution)
    last->tag = 2; // end node

  return isol;
}


void add_found_solution( BfsNode *last, Bool tag_solution)
{
  Solution *sol;

  sol = new_Solution(last, tag_solution);
  
  if (Sols_for_learning==NULL) 
    Sols_for_learning = sol;
  else if (Sols_for_learning->path_len > sol->path_len){
    Sols_for_learning = sol;
    numSolutions = 0;
    printf("\n Restarting Solution List ....previous solution discarded!");
  }
  else{
    sol->next = Sols_for_learning;
    Sols_for_learning = sol;
  }
  
  numSolutions++;
  
}

void print_solution_path(Solution *sol){
  SolPath *iNodePath;
 
  for(iNodePath=sol->path;iNodePath != NULL; iNodePath = iNodePath->next){
    printf("\n ");
    print_op_name(iNodePath->node->op);
  }
}


//The number of siblings that belongs to a best-cost solution.
float rank_node_paralelism(BfsNode *node){
  int good_siblings = 0;
  BfsNode *i_sibling;

  for (i_sibling = node->father->children; i_sibling != NULL; i_sibling = i_sibling->child_next){
    if (i_sibling->tag > 0)
      good_siblings++;
  }
  return good_siblings;
}

float rank_node_difficulty(BfsNode *node){
  int ft, i;
  int min = INFINITY;

  for (i=0; i < gef_conn[node->op].num_A; i++)
    {
      ft = gef_conn[node->op].A[i];
      
      if (LESS(gft_conn[ft].num_A, min))
	{
	  min = gft_conn[ft].num_A;
	}
      
    }
  
    return 1.0/min;
}


float rank_solution_preference(Solution *iSol, int Type){
  float node_preference, rank_value; 
  int step;
  SolPath *iSolPath;
  rank_value = 0;

  for (iSolPath = iSol->path, step=0; iSolPath!= NULL; iSolPath = iSolPath->next, step++){
    if (Type==PREFERENCE_PARALELISM)
      node_preference = rank_node_paralelism(iSolPath->node);
    else 
      node_preference = rank_node_difficulty(iSolPath->node);
    rank_value = rank_value + (node_preference * ((1.0 * (iSol->path_len - step)) / iSol->path_len));
  }  
  return rank_value;

}


int compare_ranked_sols( const void* a, const void* b ) {
  Solution ** isola = (Solution**) a; 
  Solution ** isolb = (Solution**) b; 

  if (LESS_EQ((*isola)->paralelism, (*isolb)->paralelism))
    {
      return 1;
    }
  
  if (FLOAT_EQUAL((*isola)->paralelism, (*isolb)->paralelism) &&
      LESS((*isola)->difficulty,(*isolb)->difficulty))
      return 1;
  else 
    return -1;
  
}

void print_static_literals(){

  Facts *f;
  
  printf("\nStatic Literals:");
  for ( f = gStatic_literals; f; f = f->next ) {
    printf("\n");
    print_Fact( f->fact );
  }


}



Bool action_causality (int supporter, int op){
  int i, j, add;
  Action *a = gop_conn[op].action;

  
  for (i=0; i < gef_conn[supporter].num_A; i++){
    add = gef_conn[supporter].A[i];

    for (j=0; j < a->num_preconds; j++ ){
      if (add == a->preconds[j]){
	return TRUE;
      }
    }
  }
  return FALSE;
}


Bool action_interference(int op, int affected)
{
  int i, j, del;
  Action *a = gop_conn[affected].action;

  for (i=0; i < gef_conn[op].num_D; i++)
    {
      del = gef_conn[op].D[i];
      for (j=0; j < gef_conn[affected].num_A; j++){
	if (del == gef_conn[affected].A[j])
	  return TRUE;
      }
      
      for (j=0; j < a->num_preconds; j++ ){
	if (del == a->preconds[j])
	  return TRUE;
      }
    }
  return FALSE;
}

Bool commutative_actions(int op1, int op2){
  if (!action_interference(op1,op2) && 
      !action_interference(op2,op1) &&
      !action_causality(op1,op2) && 
      !action_causality(op2,op1))
    return TRUE;
  else
    return FALSE;
}



void tag_sols_commutative_states(void){
  Solution *iSol;
  SolPath *iSolPath, *nextinPath;
  
  printf("\nComputing Commutative paths...\n");
  // The root node
  iSol = Sols_for_learning;
  //  print_solution_path(iSol);


  for (iSol = Sols_for_learning; iSol!= NULL; iSol=iSol->next){
    for (iSolPath = iSol->path; iSolPath!= NULL; iSolPath = iSolPath->next){
//       printf("\n");
//       print_op_name(iSolPath->node->op);
      
      nextinPath = iSolPath->next;
      if (nextinPath != NULL){
	if (!nextinPath->node->cm_repeated){
	  if (commutative_actions(iSolPath->node->op,nextinPath->node->op)){
// 	    printf("\n Commutative repeated (%d): ", nextinPath->node->g);
// 	    print_op_name(nextinPath->node->op);
// 	    printf("\n");
	    nextinPath->node->cm_repeated = TRUE;
	  }
	}
      }
    }
  }
}
  

  
// =====================================================================
// For drawing search tree with Dot (Graphviz)
// =====================================================================

void fprint_op_name(FILE *ex_file, int index)
{
  int i;
  Action *a = gop_conn[index].action;

  if ( !a->norm_operator &&
       !a->pseudo_action ) {
    fprintf(ex_file, "REACH-GOAL");
  } else {
    fprintf(ex_file, "%s", a->name ); 
    for ( i = 0; i < a->num_name_vars; i++ ) {
      fprintf(ex_file, " %s", gconstants[a->name_inst_table[i]]);
    }
  }
}


void data_draw_search_tree (void)
{
  FILE *draw_file;
  char filepath[MAX_LENGTH];
  gdraw_line = 1;
  gdraw_node_num = 0;

  sprintf(filepath, "%s%s", gcmd_line.path, "ff-search.dot");
  draw_file = fopen(filepath, "w");
  

  fprintf(draw_file, "digraph G { \n");
  
  draw_traverse_tree(draw_file, gSearch_tree);
  fprintf(draw_file, "} \n");

}



void draw_node_data(FILE *draw_file, BfsNode *node){
  Bool good_nodes;

  switch (gcmd_line.display_info) {
  case 300:
    good_nodes = FALSE;
    break;
  case 301:
    good_nodes = TRUE;
    break;
  case 302:
    good_nodes = FALSE;
    if (node->g > 2)
      return;
    break;
  }

  if (!good_nodes || node->tag>0){
    fprintf(draw_file, "  n%d [label=\"g:%d\\nh:%d\", shape=circle]; \n",node->number, node->g, node->h);
    if (node->tag>0)
      fprintf(draw_file, "  n%d [color=darkgreen]; \n",node->number);
    if (node->repeated)
      fprintf(draw_file, "  n%d [peripheries=2]; \n",node->number);
    if (node->cm_repeated)
      fprintf(draw_file, "  n%d [style=filled]; \n",node->number);


    fprintf(draw_file, "  n%d -> n%d [label=\"",node->father->number, node->number);
    fprintf(draw_file, "%s", gop_conn[node->op].action->name);
    fprintf(draw_file, " %s", gconstants[gop_conn[node->op].action->name_inst_table[0]]);
    //fprint_op_name(draw_file, node->op);
    
    fprintf(draw_file, "\"]; \n");
  }
}


void draw_traverse_tree(FILE *draw_file, BfsNode *node){
  BfsNode *child;
  
  if (node->father!=NULL){
    draw_node_data(draw_file, node);
  }
  else
    fprintf(draw_file, "  n0 [label=root, shape=circle]; \n");

    
  
  for (child=node->children; child!=NULL; child=child->child_next){
    child->number = ++gdraw_node_num;
    draw_traverse_tree(draw_file, child);
  }
}




/* INFO RECORD FUNCTIONS */
/* Saving search tree information for analysis */


/*Cada nodo debe tener un campo tag para ponerle el tipo de nodo
  dentro de este computo.  PONEMOS A TRUE los que pertenecen a una solución en el search tree*/
void compute_tree_h_star(){
  Solution *iSol;
  SolPath *iSolPath;
  BfsNode *iNode, *kNode;
  BfsNodeList *jNodelist;
  int h_star;

  for (iSol = Sols_for_learning; iSol!= NULL; iSol=iSol->next){
    h_star = 0;
    for (iNode = iSol->last_node; iNode != NULL; iNode = iNode->father){
      iNode->h_star = h_star++;
    }
  }
   
  if (gcmd_line.save_repeated_lists){
    for (iSol = Sols_for_learning; iSol!= NULL; iSol=iSol->next){
      for (iNode = iSol->last_node; iNode != NULL; iNode = iNode->father){
	for (jNodelist = iNode->repeated_list; jNodelist != NULL; jNodelist = jNodelist->next){
	  h_star = iNode->h_star;
	  for (kNode = jNodelist->node; kNode != NULL; kNode = kNode->father){
	    if (kNode->h_star != -1){ /* empty value*/
	      break;
	    }
	    kNode->h_star = h_star++;
	  }
	}
      }
    }
  }
}



void record_node_data(FILE *file, BfsNode *node, int type){
  int normalized_tag = 0;
  int aux_int, aux_int2;
  
  switch(type){
  case 1: /* H-LABELED */
    record_node_counter++;
    if (node->tag > 0) normalized_tag = 1;

    if (node->h_star == -1) aux_int = node->father->h_star + 1;
    else aux_int = node->h_star;
	
    if (node->father)  aux_int2 = node->father->number;
    else aux_int2 = -1;
      
    /*#no. node-number father real_tag norm_tag h_ff h-star */
	   fprintf(file, "%6ld %6ld %6ld %2d %2d %3d %3d\n", record_node_counter, node->number, aux_int2,  
		   node->tag, normalized_tag, node->h, aux_int);
    break; 
  }

}

void record_traverse_tree(FILE *file, BfsNode *node, int type, Bool only_examples){
  BfsNode *child;
  
  if (node->father!=NULL){
    record_node_data(file, node, type);
  }
  
  if (!only_examples || node->tag > 0){
    for (child=node->children; child!=NULL; child=child->child_next){
      record_traverse_tree(file, child, type, only_examples);
    }
  }
}


void record_h_labeled_nodes(void)
{
  FILE *output_file; 
  int success_closing;
  char output_file_name[MAX_LENGTH];
  int  first_time=0;
  int line_counter=0;
  char line[MAX_LENGTH] = "";

  record_node_counter = 0;
  compute_tree_h_star();
  
  sprintf(output_file_name, "%s%s", gcmd_line.knowledge_path, "h-labeled-nodes.txt");

  output_file=fopen(output_file_name, "r");
  if (output_file==NULL){
      first_time++;
      line_counter++;
  }
  else{
    while(fgets(line,sizeof(line), output_file)) 
      line_counter++;    
    fclose(output_file);
    
  }
  output_file=fopen(output_file_name, "a+");
  printf("*****************    cosas");
  if (output_file == NULL){
     printf("\ncan't open file %s to write learning results\n", output_file_name);
  }
  else {
    /* output info:*/
    if (first_time > 0 )
      {
	fprintf(output_file, "#no. node-number father real_tag norm_tag h_ff h-star\n");
      }
    
    record_traverse_tree(output_file, gSearch_tree, 1, TRUE);
    
    success_closing=fclose(output_file); 
    if (success_closing != 0){
      printf("\ncan't close file %s\n", output_file_name);
    }
  }
}
