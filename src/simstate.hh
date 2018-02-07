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
extern U8 img[HEIGHT][WIDTH][3];

//Input States
extern int cursorSize;
extern bool Rmouse; //down?
extern bool Lmouse; //down?
extern bool LShift;  //down?
extern int  lastX;  //last cursor x
extern int  lastY;  //last cursor y
extern int  action; //what action does mouse do?

enum
{
    ACT_CLEAR, //clear node
    ACT_DIRICHLET,  //add dirichlet boundary condition
    ACT_BODY  //add mixed dirichlet/neumann (robin) boundary condition
};

//Flow States (private)
typedef double F;
extern F ambientRho;
extern F ambientT;
extern F ambientVx;
extern F ambientVy;
extern F editRho;
extern F editT;
extern F editVx;
extern F editVy;

//drawing
extern F color_Tmin;
extern F color_Tmax;
extern F color_pmin;
extern F color_pmax;
extern F color_rhomin;
extern F color_rhomax;

//CFD Grid, Flow States, Boundary Conditions
//Grid Nodes shared between CPU and GPU
struct NodeState
{
    F rho; //density
    F vx;  //x-velocity
    F vy;  //y-velocity
    F T;   //temperature

    U8 type;

    enum { BODY_VARIABLE_T, BODY_FIXED_T, BODY_NO_T };
    U8 subtype;
};
extern NodeState grid[WIDTH][HEIGHT];

extern int drawType;
enum { DISP_BC, DISP_T, DISP_P, DISP_RHO };

void uploadGrid();
void downloadGrid();

void simClear();
void simFill();
void simAutoFitColorScale();
void simMouseMove(int x, int y);


void simInit();
void simDestroy();
void simStep();
void simPaint();


#endif  // ___IMAGE_HH

