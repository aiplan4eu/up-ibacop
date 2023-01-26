
/*********************************************************************
 * File: landmarks.h
 * Description: code for computing landmarks
 *
 * Author: Raquel Fuentetaja 2011
 *
 *********************************************************************/ 






#ifndef _LANDMARKS_H
#define _LANDMARKS_H

//main function for computing landmarks following  J. Hoffmann, J. Porteous and L. Sebastia method
void get_landmarks( State *S, Bool compute_also_edges );

//main function for computing landmarks by the method of deactivating facts one by one and building the RPG. 
void get_landmarks_naive( State *S );

void generate_landmarks (int time);
void insert_landmark_edge( int pre_ft, int ft);
int  initialize_landmarks( int max );
void print_landmarks (void);
void extract_landmarks( int max );
void verify_landmarks( State *S );
void remove_bad_edges(void);

#endif /* _LANDMARKS_H */


