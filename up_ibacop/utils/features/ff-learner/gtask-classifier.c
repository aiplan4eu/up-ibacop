/*
** Goal Task Classifier for ROLLER
** Tomas de la Rosa
** Started on Sept 28, 2011
*/

#include "gtask-classifier.h"


void load_tilde_gtask_tree(void)
{
  char dptree_file[MAX_LENGTH] = "";

  sprintf(dptree_file,"%s%s",gcmd_line.knowledge_path, "goal-classifier/tilde/goal-tasks.out");

  GT_tree_head =  load_tilde_derived_pred_tree(dptree_file);

  fill_predicate_and_action_indexes(GT_tree_head);

  printf("\n\nDPTREES: tree loaded %s",dptree_file);
  fflush(stdout);

  /* trace for printing all loaded trees*/
  
  if ( gcmd_line.display_info == 130 ) {
  printf("\n");
  print_LTtree(GT_tree_head,0);
  }  
}



void classify_goaltask(void){
  LTnode *nproposal;


  load_tilde_gtask_tree();
  initialize_matches(GT_tree_head);
  nproposal=match_ahead(GT_tree_head, NULL);
  printf ("\n GOAL TASK CLASS: %s", nproposal->proposal->class);
  strcat(gcmd_line.knowledge_path, nproposal->proposal->class);
  strcat(gcmd_line.knowledge_path, "/");

  printf ("\n switching to: %s", gcmd_line.knowledge_path);
//     printf ("\n\n GTASK Done!");

}

