/*
** hornclause-reader.c
** 
** Reads the TILDE Background Knowledge in the form of Horn Clauses
** 
** Date: 2011.09.16
*/


#ifndef  HORNCLAUSE_REDER_H_
# define HORNCLAUSE_REDER_H_


#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <ctype.h>
#include <sys/types.h>
#include "ff.h"


#define DPTYPE_JOIN 0
#define DPTYPE_RECURSIVE 1

#define MAX_BGCLAUSES 20

#define DPAND_MAX_LITERALS 3


#define DPSTATE_NEXT 1
#define DPSTATE_CURRENT 2
#define DPSTATE_STATIC 3


#define DPSTATE_NEXT 1
#define DPSTATE_CURRENT 2
#define SPSTATE_STATIC 3

/*for matches*/
typedef struct _DPand_literal {
  int index; 
  int factints[DPAND_MAX_LITERALS];
  int *args;
  
  int req_ints;   /* the number of facts required to be true*/
  int state_req;  /* the current computation of satisfied facts */
  Bool dp_state;  /* it is true the the dp state*/

} DPand_literal;



typedef struct _DPvar {

  char *name;
  int argindex; 

} DPvar;


typedef struct _DPvarlist {

  char *name;
  int index; 
  Bool free;
  
} DPvarlist;


typedef struct _DPmetapred {
  int type;
  int pred_index;
  char *name;
  int index;
  int nvars;
  DPvar *vars;
   
  struct _DPmetapred *next;

} DPmetapred;


  
typedef struct _DPbodymatch {
  int *argpos;
  struct _DPbodymatch *next;
} DPbodymatch;
 

typedef struct _DPclause{


  int type;                /* 0: DP_AND, 1:DP_REC: */
  char *derived_pred;      /* The name of the derived pred */
  int dp_index;            /* For identifying the derived pred */    
  DPvarlist *varlist;
  int nvars;
  DPmetapred *head;
  int nbodypreds;
  DPmetapred **bodypreds;
  DPbodymatch *bodymatch; 
  DPand_literal *and_literals;
  int num_and_literals;
  int dpstate_type;      /*Identifying when in needed to compute this*/
} DPclause;



/* Global variables */


DPclause gDP_bgclauses[MAX_BGCLAUSES];
int gDP_num_clauses;
DPmetapred *gDP_metapreds;


void parse_bg_line(char *bg_line);
void load_dp_background_clauses(const char *bg_file);
void load_tilde_dp_clauses(void);
void pruebaDP(void);

#endif 	    /* END HORNCLAUSE_READER_H */
