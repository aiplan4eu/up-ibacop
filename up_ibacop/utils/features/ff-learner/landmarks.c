/*********************************************************************
 * File: landmarks.c
 * Description: code for computing landmarks following the method in
 *              J. Hoffmann, J. Porteous and L. Sebastia (2004) 
 *              "Ordered Landmarks in Planning", Volume 22, pages 
 *              215-278
 *
 * Author: Raquel Fuentetaja 2011
 *        
 *********************************************************************/ 


#include "ff.h"
#include "output.h"
#include "relax.h"
#include "landmarks.h"


/* local globals
 */

/* for landmarks computation */
int **llandmarks_at;
int *lnum_landmarks_at;
Bool compute_edges = FALSE;

/* for trace */
Bool trace = TRUE;


// main function for computing landmarks and landmark edges
// compute_edges is a boolean indicating whether computing landmark edges or not
void get_landmarks( State *S, Bool compute_also_edges)
{
  int max;
  Bool solvable;
  
  if (compute_also_edges)
    compute_edges = TRUE;

  solvable = build_fixpoint( S, &max );

  if ( gcmd_line.display_info == 126 ) {
    print_fixpoint_result();
  }

  if ( solvable ) {
    extract_landmarks( max );
  }

  reset_fixpoint( max );
  
  //TRACE
  if (trace)
    print_landmarks(); 

  verify_landmarks( S );

  //TRACE
  if (trace)
    print_landmarks(); 

  /* restore default value */
  compute_edges = FALSE;
}



void generate_landmarks (int time )
{

  int i, j, k, l, m, ft, pre_ft, ef, other_ef;
  Bool candidate_landmark;

  
  if ( gcmd_line.display_info == 127 ) {
    printf("\nlandmarks at step %3d: ", time-1);
  }

  for ( i = 0; i < lnum_landmarks_at[time]; i++ ) {

    ft = llandmarks_at[time][i];
     
    for ( j = 0; j < gft_conn[ft].num_A; j++ ) {
      ef = gft_conn[ft].A[j];
      if ( gef_conn[ef].level != time - 1 ) continue; 

      for ( k = 0; k < gef_conn[ef].num_PC; k++ ) {
	pre_ft = gef_conn[ef].PC[k];


	for ( l = 0; l < gft_conn[ft].num_A; l++ ) {
	  other_ef = gft_conn[ft].A[l];

          if ( gef_conn[other_ef].level != time - 1 ) continue; 

	  candidate_landmark = FALSE;
	  
	  for ( m = 0; m < gef_conn[other_ef].num_PC; m++ ) {

	    if (pre_ft == gef_conn[other_ef].PC[m])
	      {
		candidate_landmark = TRUE;
		break;
	      }
	  }
	  if ( candidate_landmark  == FALSE )
	      break;
	}

	if (candidate_landmark)
	  {
	    //add pre_ft to landmarks
	    gft_conn[pre_ft].is_landmark = TRUE;
	    //add the landmark edge
	    if (compute_edges)
	      insert_landmark_edge(pre_ft,ft);
	    //add pre_ft to set of landmarks for the recursive step
	    llandmarks_at[gft_conn[pre_ft].level][lnum_landmarks_at[gft_conn[pre_ft].level]++] = pre_ft;
	  }
      }
    }
  }
}


void insert_landmark_edge( int pre_ft, int ft)
{
  int i;

  for (i = 0; i < gft_conn[pre_ft].num_landmark_edges; i++)
    {
      if (gft_conn[pre_ft].landmark_edges[i] == ft)
	return;
    }
  gft_conn[pre_ft].landmark_edges[gft_conn[pre_ft].num_landmark_edges++] = ft;
}



int initialize_landmarks( int max )

{

  static Bool first_call = TRUE;
  static int highest_seen;

  int i, max_landmark_level, ft;
 

  if ( first_call ) {
    llandmarks_at = ( int ** ) calloc( RELAXED_STEPS_DEFAULT, sizeof( int * ) );
    lnum_landmarks_at = ( int * ) calloc( RELAXED_STEPS_DEFAULT, sizeof( int ) );
     for ( i = 0; i < RELAXED_STEPS_DEFAULT; i++ ) {
      llandmarks_at[i] = ( int * ) calloc( gnum_ft_conn, sizeof( int ) );
    }
     if (compute_edges)
       {
	 for ( i = 0; i < gnum_ft_conn ; i++ ) {
	   gft_conn[i].landmark_edges =  ( int * ) calloc( gnum_ft_conn, sizeof( int ) );
	 }
       }

    highest_seen = RELAXED_STEPS_DEFAULT;
    first_call = FALSE;
  }

  if ( max + 1 > highest_seen ) {
    for ( i = 0; i < highest_seen; i++ ) {
      free( llandmarks_at[i] );
    }
    free( llandmarks_at );
    free( lnum_landmarks_at );
    highest_seen = max + 10;
    llandmarks_at = ( int ** ) calloc( highest_seen, sizeof( int * ) );
    lnum_landmarks_at = ( int * ) calloc( highest_seen, sizeof( int ) );
    for ( i = 0; i < highest_seen; i++ ) {
      llandmarks_at[i] = ( int * ) calloc( gnum_ft_conn, sizeof( int ) );
    }
  }

  for ( i = 0; i < max + 1; i++ ) {
    lnum_landmarks_at[i] = 0;
  }

  for (i = 0; i < gnum_ft_conn; i++)
    {
     gft_conn[i].is_landmark = FALSE;
     gft_conn[i].num_landmark_edges = 0;  
    }

  max_landmark_level = 0;
  for ( i = 0; i < gnum_flogic_goal; i++ ) {
    ft = gflogic_goal[i];
    if ( gft_conn[ft].level > max_landmark_level ) {
      max_landmark_level = gft_conn[ft].level;
    }
    llandmarks_at[gft_conn[ft].level][lnum_landmarks_at[gft_conn[ft].level]++] = ft;
    gft_conn[ft].is_landmark = TRUE;
  }

  activate_all();
  return max_landmark_level;
}

void print_landmarks (void)
{
  int i, j;
  

  for (i = 0; i < gnum_ft_conn; i++)
    {
      if ( gft_conn[i].is_landmark )
	{
	  printf("\n\nLANDMARK: ");
	  print_ft_name(i);
	
	  if (compute_edges)
	    {
	      printf("\nEDGES:\n");
	      for (j = 0; j <  gft_conn[i].num_landmark_edges; j++)
		{
		  print_ft_name(i);
		  printf("-->");
		  print_ft_name(gft_conn[i].landmark_edges[j]);
		}
	    }
	  
	}
    }
  fflush(stdout);
}



void extract_landmarks( int max )

{

  int  max_landmark_level, time;

  max_landmark_level = initialize_landmarks( max );

  for ( time = max_landmark_level; time > 0; time-- ) {
    generate_landmarks( time );
  }

}




void verify_landmarks (State *S)
{
  int i, j,  max, ef;
  Bool solvable, one_deactivated, in_initial_state;
  
  //TRACE
  if (trace)
     printf("\n\nverifying landmarks...\n");
  
  for (i = 0; i < gnum_ft_conn; i++)
    {
    // for each landmark
    if (gft_conn[i].is_landmark)
      {
	one_deactivated = FALSE;
	in_initial_state = FALSE;
	
	//TRACE
	if (trace)
	  {
	  printf("\n");
	  print_ft_name (i);
	  fflush(stdout);
	  }
		
	for ( j = 0; j < S->num_F; j++ ) {
	  if (S->F[j] == i)
	    in_initial_state = TRUE;
	}

	if (in_initial_state)
	  continue;

        for (j = 0; j < gft_conn[i].num_A ; j++ ){
	  ef = gft_conn[i].A[j];
	
	  // for each action adding the landmark deactivate the action		
	  gop_conn[gef_conn[ef].op].RP_deactivated = TRUE;
	
	  //TRACE
	  if (trace)
	    {
	      printf("\ndeactivate ");
	      print_op_name(gef_conn[ef].op);
	      fflush(stdout);
	    }

	  one_deactivated = TRUE;
	}

	if (one_deactivated)
	  {
	    // build the RPG
	    solvable = build_fixpoint( S, &max );

	    if (solvable)
	      {
		//TRACE
		if (trace)
		  {
		    printf("\neliminating...");
		    print_ft_name(i);
		  }
		
		// if solvable the fact is not a landmark
		gft_conn[i].is_landmark = FALSE;
		// remove edges starting on the fact
		gft_conn[i].num_landmark_edges = 0;
	      }
	    activate_all();
            reset_fixpoint( max );
	  }
      }
    }
  if (compute_edges)
    remove_bad_edges();
}


void remove_bad_edges(void)
{

  int *new_edges;
  int i, j, num_edges;
  Bool one_removed;

  for (i = 0; i < gnum_ft_conn; i++)
    {
    // for each landmark
    if (gft_conn[i].is_landmark)
      {
	num_edges = 0;
	one_removed = FALSE;
	// for each edge
        new_edges =  ( int * ) calloc( gnum_ft_conn, sizeof( int ) );

	for (j = 0; j < gft_conn[i].num_landmark_edges; j++)
	  {
	    if (!gft_conn[gft_conn[i].landmark_edges[j]].is_landmark)
	      {
		one_removed = TRUE;
	      }
	    else
	      {
		new_edges[num_edges++]=gft_conn[i].landmark_edges[j];
	      }
	  }
	if (one_removed)
	  {
	    free (gft_conn[i].landmark_edges);
	    gft_conn[i].landmark_edges = new_edges;
	    gft_conn[i].num_landmark_edges = num_edges;
	  }
	else
	  {
	    free (new_edges);
	  }
      }
    }
}



/* Next function compute landmarks by deactivating facts one by one:
1. solve the relaxed task in order to guarantee it is solvable.
2- for each fact f {deactivate f and solve the relaxed task, if it is not solvable, f is a landmark} 
 */
void get_landmarks_naive( State *S)
{
  int max, i;
  Bool solvable;
  
  solvable = build_fixpoint( S, &max );

  if ( gcmd_line.display_info == 126 ) {
    print_fixpoint_result();
  }

  if ( solvable ) {

    /* Initialize landmarks */
    for (i = 0; i < gnum_ft_conn; i++)
      {
	gft_conn[i].is_landmark = FALSE;
	gft_conn[i].num_landmark_edges = 0;  
	// by default facts are not deactivated, so it is nothing to do here
      }

    for  ( i = 0; i < gnum_ft_conn; i++ )
      {
	reset_fixpoint( max );

	gft_conn[i].RP_deactivated= TRUE;
      
	solvable = build_fixpoint( S, &max );

	if (!solvable)
	  {
	    gft_conn[i].is_landmark = TRUE;
	  }

	gft_conn[i].RP_deactivated= FALSE;

      }
  }

  reset_fixpoint( max );
  
  //TRACE
  if (trace)
    print_landmarks(); 

}
