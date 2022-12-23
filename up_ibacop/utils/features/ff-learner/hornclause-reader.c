/*
** hornclause-reader.c
** 
** Reads the TILDE Background Knowledge in the form of Horn Clauses
** 
** Date: 2011.09.16
*/


#include "hornclause-reader.h"
#include "derived-preds.h"
#include "output.h"
#include <string.h>
#include "memory.h"
#include "matcher.h"


void print_DPmetapred(DPmetapred *mpred){
  int i;

  printf("\n [%d][%d]%s(",mpred->type, mpred->pred_index, mpred->name);
  for (i=0 ; i< mpred->nvars; i++){
    if (i > 0) printf(",");
    printf("%s[%d]", mpred->vars[i].name, mpred->vars[i].argindex);
  }
  printf(")");
}

void print_DPclause (DPclause *clause){
  DPmetapred *i_mpred;
  DPbodymatch *i_match;
  int i,j;
  
  printf("\n[BG-Clause}: Vars[");
  for (i=0 ; i< clause->nvars; i++){
    if (i == 0){
      if (clause->varlist[i].free == FALSE) 
	printf("-");
    printf("%s", clause->varlist[i].name);
    }
    else{
      if (clause->varlist[i].free == FALSE) 
	printf("-");
      printf(",%s", clause->varlist[i].name);
    }
  }
  
  printf("[");
  for (i_match = clause->bodymatch; i_match != NULL; i_match = i_match->next){
    printf("(");
    for (i = 0; i< clause->nbodypreds; i++)
      printf ("%d", i_match->argpos[i]);
    printf(")");
  }
  printf("]");
  
  printf("]\n Head:");
  print_DPmetapred(clause->head);
  printf("\n Body:");
  
  for (i = 0; i< clause->nbodypreds; i++){
    i_mpred = clause->bodypreds[i];
    print_DPmetapred(i_mpred);
    if (i + 1 < clause->nbodypreds)
      printf(",");
  }
  
  for (i = 0; i< clause->num_and_literals; i++){
    printf("\n %s(", clause->head->name);
    for (j = 0; j < clause->nvars; j++){
      if (j>0) 
	printf(",");
        printf("%s", gconstants[clause->and_literals[i].args[j]]);
    }
    printf(")");
    
  }
}


DPmetapred * new_metapred(char *strpred, char *strargs, Bool derived){
  DPmetapred *metapred;
  char var_names[MAX_VARS][MAX_LENGTH];
  int nvars,i;
  DPvar *result_vars;
  char *tkargs;
  char name_without_hyphens[MAX_LENGTH];


  //printf ("\n >> NEW Metapred: %s %s", strpred, strargs);

  metapred = (DPmetapred *) calloc(1, sizeof(DPmetapred));
  metapred->name = (char*)calloc(strlen(strpred), sizeof(char));
  strcpy(metapred->name,strpred);
  
  nvars=0;
  tkargs = strtok(strargs,",");
  
  while(tkargs != NULL){
    strcpy(var_names[nvars],tkargs); 
    nvars ++;
    tkargs=strtok(NULL,",");
  }

  result_vars = (DPvar *) calloc(nvars,sizeof(DPvar));  
  
  for(i=0; i<nvars; i++){
    result_vars[i].name = (char *) calloc (strlen(var_names[i])+1,sizeof(char));
    strcpy(result_vars[i].name, var_names[i]);
    result_vars[i].argindex = -1;
  }

  metapred->vars = result_vars;
  metapred->nvars = nvars;

  /*filling the predicate index and type */
  if (!derived){
   
    if (strstr(metapred->name, "dpand_") != NULL)
      metapred->type = PREDICATETYPE_DERIVED;
    else if (strstr(metapred->name, "static_fact_") != NULL)
      metapred->type = PREDICATETYPE_STATIC;
    else if (strstr(metapred->name, "state_") != NULL)
      metapred->type = PREDICATETYPE_STATE;
    else if (strstr(metapred->name, "next_") != NULL)
      metapred->type = PREDICATETYPE_NEXT;
    else if (strstr(metapred->name, "target_goal_") != NULL)
      metapred->type = PREDICATETYPE_TARGET;
    else if (strstr(metapred->name, "achieved_") != NULL)
      metapred->type = PREDICATETYPE_ACHIEVEDGOAL;
    else if (strstr(metapred->name, "goal_") != NULL)
      metapred->type = PREDICATETYPE_GOAL;
    
    if (metapred->type == PREDICATETYPE_DERIVED){
      for (i = 1; i < gDP_num_clauses; i++){
	if (strcmp(gDP_bgclauses[i].head->name, metapred->name) == SAME){
	  metapred->pred_index = i;
	  break;
	}
      }
    }
    else
      {
	for (i = 1; i < gnum_orig_predicates; i++)
	  {
	    name_without_hyphens[0] = '_';
	    strcpy(&(name_without_hyphens[1]),gpredicates[i]);
	    string_to_lower(name_without_hyphens,name_without_hyphens);
	    string_replace(name_without_hyphens,"-","_");
	    
	    if (strstr(metapred->name, name_without_hyphens) != NULL)
	      {
		metapred->pred_index = i;
		break;
	    }
	  }
      }

  }
    
  return metapred;
}


 

void parse_bg_line(char *bg_line){
  string_replace(bg_line,"(", "|");
  string_replace(bg_line,")", ",|");
  string_replace(bg_line," ", ""); 
  return;
}


void split_compound_and_clause(DPclause *clause, int ipred){
  int i,j, parent_clause; 
  DPmetapred **newbodypreds;
  int n_newbodypreds = 0;

  printf ("\n Spliting %s", clause->head->name);
  newbodypreds = (DPmetapred **) calloc(clause->nbodypreds + 1,sizeof(DPmetapred *));
  for(i=0; i < clause->nbodypreds; i++){
    if (i == ipred){
      parent_clause = clause->bodypreds[i]->pred_index;
      for (j = 0; j < gDP_bgclauses[parent_clause].nbodypreds; j++){
	newbodypreds[n_newbodypreds++] = gDP_bgclauses[parent_clause].bodypreds[j];
      }
    }
    else{
      newbodypreds[n_newbodypreds++] = clause->bodypreds[i];
    }
  }
  
  clause->nbodypreds = n_newbodypreds;
  clause->bodypreds = newbodypreds;

}


void propagate_compound_and_clauses(void){
  int i,j;

  printf ("\n Propagating compound clauses...");
  for (i=0; i< gDP_num_clauses; i++){
    for (j=0; j < gDP_bgclauses[i].nbodypreds; j++){
      if (gDP_bgclauses[i].bodypreds[j]->type == PREDICATETYPE_DERIVED){
	split_compound_and_clause(&(gDP_bgclauses[i]),j);
	j--;
      }
    }
  }

}

void load_dp_background_clauses(const char *bg_file){
  char line[MAX_LENGTH] = "";
  FILE *fp;
  char *tokens, *strargs;
  DPmetapred *last_metapred;
  Bool pending_clause;
  int nbodypreds;
  
  gDP_metapreds = NULL;
  gDP_num_clauses = 0;

  if((fp = fopen(bg_file, "r")) == NULL){
    printf("\nDPTREES: Can't find bg file: %s\n", bg_file);
    printf("\n         Planning with trees without Derived Predicates!\n\n");
    return;
  }
  
  while(fgets(line,sizeof(line), fp)){    
    if(strstr(line,":-")!=NULL){

      parse_bg_line(line);
      
      tokens = strtok(line,"|");
//       printf ("pred: %s \n", tokens);
      
      strargs = strtok(NULL,"|");
//       printf ("args: %s \n", strargs);
      
      if (gDP_metapreds != NULL){
	last_metapred->next = new_metapred(tokens, strargs, TRUE);
	last_metapred = last_metapred-> next;
      }
      else{
	gDP_metapreds = new_metapred(tokens, strargs, TRUE);
	last_metapred = gDP_metapreds;
      }
      
//       new_clause = (DPclause *) calloc(1,sizeof(DPclause));
      gDP_bgclauses[gDP_num_clauses].head = last_metapred;
      gDP_bgclauses[gDP_num_clauses].bodypreds = (DPmetapred **) calloc(2,sizeof(DPmetapred *));
      nbodypreds = 0;
      pending_clause = TRUE;
      /* asigning the new clause */
      
    }
    else if((strstr(line,"),")!=NULL) || (strstr(line,").")!=NULL)){	
      /*printf("\parsing body...");*/
      if (strstr(line,".")!=NULL)
	pending_clause = FALSE;


      parse_bg_line(line);

      tokens = strtok(line,"|");
      strargs = strtok(NULL,"|");
      last_metapred->next = new_metapred(tokens, strargs, FALSE);
      last_metapred = last_metapred->next;
      
      gDP_bgclauses[gDP_num_clauses].bodypreds[nbodypreds] = last_metapred;
      nbodypreds++;

	
      if (!pending_clause){
	gDP_bgclauses[gDP_num_clauses].nbodypreds = nbodypreds;
	gDP_num_clauses++;	

      }
      
    }
    
  }
  
  
  fclose(fp);
  
}


void compute_argindex(DPclause *clause){
  int i,j,k;
  DPmetapred *jMetapred;
  
  for (i=0; i < clause->nvars; i++){
    for (j=0; j < clause->nbodypreds; j++){
      jMetapred = clause->bodypreds[j];
      
      for (k = 0; k < jMetapred->nvars; k++){
	if (strstr(clause->varlist[i].name, jMetapred->vars[k].name) != NULL){
	  jMetapred->vars[k].argindex = i;
	}
      }
    }
  }
}



void compute_commonvars(DPclause *clause){
  int i,j,k;
  DPmetapred *jMetapred;
  DPbodymatch *newcommon,*current_bmatch;
  
  clause->bodymatch = NULL;
  newcommon = (DPbodymatch *) calloc (1,sizeof(DPbodymatch));
  newcommon->argpos = (int *) calloc (clause->nbodypreds, sizeof(int));
  
  
  //   printf ("\n >> Computing common vars...");
  for (i=0; i < clause->nvars; i++){
    for (j=0; j < clause->nbodypreds; j++){
      jMetapred = clause->bodypreds[j];
      
      for (k = 0; k < jMetapred->nvars; k++){
	if (strstr(clause->varlist[i].name, jMetapred->vars[k].name) != NULL){
	  newcommon->argpos[j] = k;
	  jMetapred->vars[k].argindex = i;
	  break;
	}
      }
      
      if (k == jMetapred->nvars) /* the var was not found */
	break;
    }
    
    if (j == clause->nbodypreds) /* the var is in all metapreds */
      {
	/*printf ("\n Common Var: %s", clause->varlist[i].name);*/
	
	if (clause->bodymatch == NULL){
	  clause->bodymatch = newcommon;
	  current_bmatch = newcommon;
	}
	else{
	  current_bmatch->next = newcommon;
	  current_bmatch = current_bmatch->next;
	}

	newcommon = (DPbodymatch *) calloc (1,sizeof(DPbodymatch));
	newcommon->argpos = (int *) calloc (clause->nbodypreds, sizeof(int));
      }
  }
}

  

void compute_dpvarlist(DPclause *clause){
  int i,j,k;
  DPvarlist varlist[MAX_VARS];
  int var_index;
  DPmetapred *iMetapred;
  
  /* add heder values*/
  var_index = 0;
  for (j=0; j < clause->head->nvars; j++){
    iMetapred = clause->head;
    varlist[var_index].name = (char *) calloc(strlen(iMetapred->vars[j].name), sizeof(char));
    strcpy(varlist[var_index].name, iMetapred->vars[j].name);
    var_index++;
    }
  
  for (i = 0; i< clause->nbodypreds; i++){
    iMetapred = clause->bodypreds[i];
    for (j=0; j < iMetapred->nvars; j++){
      for (k = 0; k < var_index; k++){
	if (strstr(iMetapred->vars[j].name,varlist[k].name) != NULL) break;
      }
	
      if (k == var_index){ /* the var wasnt found*/
	varlist[var_index].name = (char *) calloc(strlen(iMetapred->vars[j].name)+1, sizeof(char));
// 	printf ("\n >>> %d", (int) strlen(iMetapred->vars[j].name));
	strcpy(varlist[var_index].name, iMetapred->vars[j].name);
	var_index++;
      }
    
    }
    
  }
  
  /* allocate this in the clause*/
  clause->nvars = var_index;
  clause->varlist = (DPvarlist *) calloc (var_index,sizeof(DPvarlist));
  for (i=0; i< var_index; i++){
    clause->varlist[i] = varlist[i];
  }
}

void load_tilde_dp_clauses(void){
    //  char filename[MAX_LENGTH] = "/home/tomas/sayphi/domains/spanner/dp-trees/derived-pred/planning-episodes.bg\0";
  char filename[MAX_LENGTH] = "";
  int i;

  sprintf(filename,"%s%s",gcmd_line.knowledge_path, "planning-episodes.bg");
  load_dp_background_clauses(filename);
//   propagate_compound_and_clauses();

  for (i=0; i< gDP_num_clauses; i++){
    compute_dpvarlist(&gDP_bgclauses[i]);
    compute_argindex(&gDP_bgclauses[i]);
    compute_commonvars(&gDP_bgclauses[i]);
    compute_derived_literals(&(gDP_bgclauses[i]));
    initialize_DPclause_states();
    
    if ( gcmd_line.display_info == 130 ) {
      print_DPclause(&gDP_bgclauses[i]);
    }


  }
  printf("\nDPTREES: BG Clauses loaded!");
}


void pruebaDP(void){
  //  char filename[MAX_LENGTH] = "/home/tomas/sayphi/domains/spanner/dp-trees/derived-pred/planning-episodes.bg\0";
  char filename[MAX_LENGTH] = "/home/tomas/sayphi/domains/satellite/dp-trees/derived-fq/planning-episodes.bg\0";
  int i;

  load_dp_background_clauses(filename);

  printf("\n Done.");
  printf("\n Número de cláusulas: %d", gDP_num_clauses);
  for (i=0; i< gDP_num_clauses; i++){
    compute_dpvarlist(&gDP_bgclauses[i]);
    compute_argindex(&gDP_bgclauses[i]);
    compute_commonvars(&gDP_bgclauses[i]);
    compute_derived_literals(&(gDP_bgclauses[i]));

    printf("\n");
    
    initialize_DPclause_states();
    print_DPclause(&gDP_bgclauses[i]);
    
  }
    
  
}
