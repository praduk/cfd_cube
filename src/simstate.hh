/*
* simstate.hh
* 
* Copyright (c) 2018 Pradu Kannan. All rights reserved.
*/

#ifndef ___SIMSTATE_HH
#define ___SIMSTATE_HH

#include "defs.hh"

//Image State
typedef unsigned char U8;
extern U8 image[HEIGHT][WIDTH][3];

//Input States
extern U8 cursorSize;
extern bool Rmouse; //down?
extern bool Lmouse; //down?
extern int lastX;  //last cursor x
extern int lastY;  //last cursor y
extern int action; //what action does mouse do?
enum
{
    ACT_CLEAR, //clear node
    ACT_BC     //add boundary condition
};

void simClear();
void simMouseMove(int x, int y);


void simInit();
void simDestroy();
void simStep();


#endif  // ___IMAGE_HH

