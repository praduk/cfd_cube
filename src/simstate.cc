/*
* simstate.cc
* 
* Copyright (c) 2018 Pradu Kannan. All rights reserved.
*/

#include "simstate.hh"
#include <stdio.h>

U8 image[HEIGHT][WIDTH][3];


U8 cursorSize;
bool Lmouse;
bool Rmouse;
int lastX;
int lastY;
int action;

void simEvaluateAction(int x, int y, int action)
{
    switch(action)
    {
        case ACT_CLEAR:
        {
            image[y][x][0] = 255;
            image[y][x][1] = 255;
            image[y][x][2] = 255;
            break;
        }
        case ACT_BC:
        {
            image[y][x][0] = 0;
            image[y][x][1] = 0;
            image[y][x][2] = 0;
            break;
        }
    }
}

void simClear()
{
    for(int i=0; i<WIDTH; i++)
        for( int j=0; j<HEIGHT; j++)
            simEvaluateAction(i,j,ACT_CLEAR);
}

void simMouseMove(int x, int y)
{
    lastX = x;
    lastY = y;

    if( Lmouse || Rmouse )
    {
        int xmin = x-cursorSize;
        int xmax = x+cursorSize;
        int ymin = y-cursorSize;
        int ymax = y+cursorSize;

        if( xmin < 0 ) xmin = 0;
        if( ymin < 0 ) ymin = 0;
        if( xmax >= WIDTH  ) xmin = WIDTH-1;
        if( ymax >= HEIGHT ) ymin = HEIGHT-1;

        if( Rmouse )
        {
            for(int i=xmin; i<=xmax; i++)
                for( int j=ymin; j<=ymax; j++)
                    simEvaluateAction(i,j,ACT_CLEAR);
        }
        else if( Lmouse )
        {
            for(int i=xmin; i<=xmax; i++)
                for( int j=ymin; j<=ymax; j++)
                    simEvaluateAction(i,j,action);
        }
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
    action = ACT_BC;
}

void simDestroy()
{
}

void simStep()
{
}
