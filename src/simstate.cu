/*
* simstate.cc
* 
* Copyright (c) 2018 Pradu Kannan. All rights reserved.
*/

#include "simstate.hh"
#include <stdio.h>

U8 image[HEIGHT][WIDTH][3];

int cursorSize;
bool Lmouse;
bool Rmouse;
bool LShift;
int lastX;
int lastY;
int action;

//Input Dirichlet Flow Variables
F ambientRho;
F ambientT;
F ambientVx;
F ambientVy;
F editRho;
F editT;
F editVx;
F editVy;

//CFD Grid, Flow States, Boundary Conditions
//Grid Nodes shared between CPU and GPU
struct NodeState
{
    F rho; //density
    F vx;  //x-velocity
    F vy;  //y-velocity
    F h;   //enthalpy

    enum{ FREE, DIRICHLET, BODY };
    int type;
};

//GPU only Nodes
struct NodeCalc
{

    //Flow State Variables
    F rho; //density
    F vx;  //x-velocity
    F vy;  //y-velocity
    //F h;   //enthalpy

    //Flow State Derivatives
    F drhodt; //density
    F drhodx; //density
    F drhody; //density
    F dvxdt;  //x-velocity
    F dvxdx;  //x-velocity
    F dvxdy;  //x-velocity
    F dvydt;  //y-velocity
    F dvydx;  //y-velocity
    F dvydy;  //y-velocity
    //F dhdt;   //enthalpy
    //F dhdx;   //enthalpy
    //F dhdy;   //enthalpy

    //First Pass Variables
    //F p;   //pressure
    //F T;   //Temperature
};

NodeState (*grid)[HEIGHT];
NodeCalc  (*gridc)[HEIGHT];

void simEvaluateAction(int x, int y, int action)
{
    switch(action)
    {
        case ACT_CLEAR:
        {
            grid[x][y].type = NodeState::FREE;
            grid[x][y].rho = ambientRho;
            grid[x][y].vx = ambientVx;
            grid[x][y].vy = ambientVy;

            image[y][x][0] = 255;
            image[y][x][1] = 255;
            image[y][x][2] = 255;
            break;
        }
        case ACT_BODY:
        {
            grid[x][y].type = NodeState::BODY;
            grid[x][y].rho = ambientRho;
            grid[x][y].vx = 0.0;
            grid[x][y].vy = 0.0;
            image[y][x][0] = 0;
            image[y][x][1] = 0;
            image[y][x][2] = 0;
            break;
        }
        case ACT_DIRICHLET:
        {
            grid[x][y].type = NodeState::DIRICHLET;
            grid[x][y].rho = ambientRho;
            grid[x][y].vx = 0.0;
            grid[x][y].vy = 0.0;
            image[y][x][0] = 0;
            image[y][x][1] = 0;
            image[y][x][2] = 0;
            break;
        }
    }
}

void simEvaluateActionOnLine(int x0, int y0, int x1, int y1, int action)
{
    int dx = abs(x1-x0), sx = x0<x1 ? 1 : -1;
    int dy = abs(y1-y0), sy = y0<y1 ? 1 : -1; 
    int err = (dx>dy ? dx : -dy)/2, e2;

    for(;;)
    {
        simEvaluateAction(x0,y0,action);
        if (x0==x1 && y0==y1) break;
        e2 = err;
        if (e2 >-dx) { err -= dy; x0 += sx; }
        if (e2 < dy) { err += dx; y0 += sy; }
    }
}

void simClear()
{
    for(int i=0; i<WIDTH; i++)
        for( int j=0; j<HEIGHT; j++)
            simEvaluateAction(i,j,ACT_CLEAR);
}

void simFill()
{
    for(int i=0; i<WIDTH; i++)
        for( int j=0; j<HEIGHT; j++)
            simEvaluateAction(i,j,ACT_BODY);
}

void simPaint()
{
}

void simMouseMove(int x, int y)
{
    if( Lmouse || Rmouse )
    {
        //Calculate bounds
        int xmin = x-cursorSize;
        int xmax = x+cursorSize;
        int ymin = y-cursorSize;
        int ymax = y+cursorSize;
        int lxmin = lastX-cursorSize;
        int lxmax = lastX+cursorSize;
        int lymin = lastY-cursorSize;
        int lymax = lastY+cursorSize;

        if( xmin < 0 )       xmin = 0;
        if( ymin < 0 )       ymin = 0;
        if( xmax >= WIDTH  ) xmax = WIDTH-1;
        if( ymax >= HEIGHT ) ymax = HEIGHT-1;
        if( lxmin < 0 )       lxmin = 0;
        if( lymin < 0 )       lymin = 0;
        if( lxmax >= WIDTH  ) lxmax = WIDTH-1;
        if( lymax >= HEIGHT ) lymax = HEIGHT-1;

        //convert to offsets
        xmin = xmin-x;
        xmax = xmax-x;
        ymin = ymin-y;
        ymax = ymax-y;
        lxmin = lxmin-lastX;
        lxmax = lxmax-lastX;
        lymin = lymin-lastY;
        lymax = lymax-lastY;

        xmin = xmin>lxmin?xmin:lxmin;
        xmax = xmax<lxmax?xmax:lxmax;
        ymin = ymin>lymin?ymin:lymin;
        ymax = ymax<lymax?ymax:lymax;

        int cs = cursorSize*cursorSize;

        if( Rmouse )
        {
            for(int i=xmin; i<=xmax; i++)
                for( int j=ymin; j<=ymax; j++)
                    if(i*i+j*j<=cs)
                        //simEvaluateAction(x+i,y+j,ACT_CLEAR);
                        simEvaluateActionOnLine(lastX+i,lastY+j,x+i,y+j,ACT_CLEAR);
        }
        else if( Lmouse )
        {
            for(int i=xmin; i<=xmax; i++)
                for( int j=ymin; j<=ymax; j++)
                    if(i*i+j*j<=cs)
                        //simEvaluateAction(x+i,y+j,action);
                        simEvaluateActionOnLine(lastX+i,lastY+j,x+i,y+j,action);
        }
        if(LShift)
        {
            lastX = x;
            lastY = y;
        }
    }
    if( !LShift )
    {
        lastX = x;
        lastY = y;
    }
}

void simInit()
{
    for(int i=0; i<HEIGHT; i++)
        for(int j=0; j<WIDTH; j++)
            for(int k=0; k<3; k++)
                image[i][j][k] = 255;
    cursorSize = 0;

    Rmouse = false;
    Lmouse = false;
    lastX = 0;
    lastY = 0;
    action = ACT_BODY;

    //Initialize Grid for Sea-Level Conditions
    cudaMallocManaged(&grid,WIDTH*HEIGHT*sizeof(NodeState)); //Shared GPU/CPU
    cudaMalloc(&gridc,WIDTH*HEIGHT*sizeof(NodeCalc)); //GPU Only

    ambientRho = editRho = 1.225;
    ambientT = editT = 288.15;
    ambientVx = editVx = 0.0;
    ambientVy = editVy = 0.0;
}

void simDestroy()
{
    cudaFree(grid);
}

void simStep()
{
}
