/*
* simstate.cc
* 
* Copyright (c) 2018 Pradu Kannan. All rights reserved.
*/

#include "simstate.hh"
#include <stdio.h>

#undef NDEBUG
#include <assert.h>

//#define CHECK(ans)
#ifndef CHECK
#define CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#endif

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

U8 img[HEIGHT][WIDTH][3]; //CPU
//U8 (*cimg)[3]; //GPU
//__device__ U8 (*cimg)[WIDTH][3]; //GPU
__device__ U8 cimg[HEIGHT][WIDTH][3]; //GPU

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

//Input Colorbar Values
F color_Tmin;
F color_Tmax;
F color_pmin;
F color_pmax;
F color_rhomin;
F color_rhomax;

__constant__ F ccolor_Tmin;
__constant__ F ccolor_Tmax;
__constant__ F ccolor_pmin;
__constant__ F ccolor_pmax;
__constant__ F ccolor_rhomin;
__constant__ F ccolor_rhomax;

int drawType;

#define cdt 1E-5
#define PixelSize 1E-1
#define invPixelSize (1.0/PixelSize)

NodeState grid[WIDTH][HEIGHT];
__device__ NodeState cgrid[WIDTH][HEIGHT];
__device__ NodeState old_cgrid[WIDTH][HEIGHT];

//GPU only Nodes
struct NodeCalc
{
    //Flow State Variables
    F p;      //pressure
    F vmx;    //mean velocity
    F vmy;    //mean velocity

    //Flow State Derivatives
    F drhodx; //density
    F drhody; //density
    F dvxdx;  //x-velocity
    F dvxdy;  //x-velocity
    F dvydx;  //y-velocity
    F dvydy;  //y-velocity
    F dpdx;   //pressure
    F dpdy;   //pressure
    F dTdx;   //temperature
    F dTdy;   //temperature
    F sigmaxx; //Cauchy stress tensor
    F sigmayy; //Cauchy stress tensor
    F sigmaxy; //Cauchy stress tensor
};

 __device__  NodeCalc cgrido[WIDTH][HEIGHT];

void uploadGrid()
{
    //void* devPtr;
    //CHECK(cudaGetSymbolAddress(&devPtr,cgrid));
    CHECK( cudaMemcpyToSymbol(cgrid,grid, TOTALNODES*sizeof(NodeState),0, cudaMemcpyHostToDevice) );
}

void downloadGrid()
{
    CHECK( cudaMemcpyFromSymbol(grid,cgrid, TOTALNODES*sizeof(NodeState),0, cudaMemcpyDeviceToHost) );
}

void simEvaluateAction(int x, int y, int action)
{
    switch(action)
    {
        case ACT_CLEAR:
        {
            grid[x][y].type = FREE;
            grid[x][y].rho = ambientRho;
            grid[x][y].vx = ambientVx;
            grid[x][y].vy = ambientVy;
            grid[x][y].T = ambientT;
            break;
        }
        case ACT_BODY:
        {
            grid[x][y].type = BODY;
            grid[x][y].rho = ambientRho;
            grid[x][y].vx = 0.0;
            grid[x][y].vy = 0.0;
            grid[x][y].T = ambientT;
            break;
        }
        case ACT_DIRICHLET:
        {
            grid[x][y].type = DIRICHLET;
            grid[x][y].rho = editRho;
            grid[x][y].vx = editVx;
            grid[x][y].vy = editVy;
            grid[x][y].T = editT;
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
                img[i][j][k] = 255;
    cursorSize = 0;

    Rmouse = false;
    Lmouse = false;
    lastX = 0;
    lastY = 0;
    action = ACT_BODY;
    drawType = DISP_BC;

    //Initialize GPU memory for Grid and set to Sea-Level Conditions
    //cudaMalloc((void**)&cgrid,WIDTH*HEIGHT*sizeof(NodeState));
    //cudaMalloc((void**)&cgrido,WIDTH*HEIGHT*sizeof(NodeCalc));
    //cudaMalloc((void**)&cimg,WIDTH*HEIGHT*24);

    ambientRho = editRho = (F)1.225;
    ambientT   = editT = (F)288.15;
    ambientVx  = editVx = (F)0.0;
    ambientVy  = editVy = (F)0.0;

    color_Tmin = 273.13;
    color_Tmax = 373.15;
    color_pmin = 50000;
    color_pmax = 150000;
    color_rhomin = 1.1;
    color_rhomax = 1.3;

    simClear();
    uploadGrid();
}

void simDestroy()
{
    cudaFree(grid);
}


/*************** DRAWING *******************/

void simAutoFitColorScale()
{
    F rhoMin = 1E99;
    F rhoMax = 0;
    F TMin = 1E99;
    F TMax = 0;
    F pMin = 1E99;
    F pMax = 0;
    for(int x=0;x<WIDTH;x++)
        for(int y=0;y<HEIGHT;y++)
        {
            F p = grid[x][y].rho*R_AIR*grid[x][y].T;
            if(grid[x][y].rho < rhoMin) rhoMin=grid[x][y].rho;
            if(grid[x][y].rho > rhoMax) rhoMax=grid[x][y].rho;
            if(grid[x][y].T < TMin) TMin=grid[x][y].T;
            if(grid[x][y].T > TMax) TMax=grid[x][y].T;
            if(p < pMin) pMin=p;
            if(p > pMax) pMax=p;
        }

    color_Tmin   = TMin - 1.0;
    color_Tmax   = TMax + 1.0;
    color_pmin   = pMin - 1.0;
    color_pmax   = pMax + 1.0;
    color_rhomin = rhoMin - 1E-12;
    color_rhomax = rhoMax + 1E-12;
}

__global__ void generateImageTemp();
__global__ void generateImagePres();
__global__ void generateImageRho();
__global__ void generateImageBC();
void simPaint()
{
    switch(drawType)
    {
        case DISP_BC: generateImageBC<<<CB,CT>>>(); break;
        case DISP_T: generateImageTemp<<<CB,CT>>>(); break;
        case DISP_P: generateImagePres<<<CB,CT>>>(); break;
        case DISP_RHO: generateImageRho<<<CB,CT>>>(); break;
    }
    CHECK( cudaMemcpyFromSymbol(img,cimg,TOTALNODES*3,0, cudaMemcpyDeviceToHost) );
}

__global__ void generateImageBC()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //Thread Index
    int tot = blockDim.x * gridDim.x; //Total Number of Threads
    for (int i=idx; i<TOTALNODES; i+=tot)
    {
        int y = i/WIDTH;
        int x = i-y*WIDTH;
        switch( cgrid[x][y].type )
        {
        case FREE:
            cimg[y][x][0] = 255;
            cimg[y][x][1] = 255;
            cimg[y][x][2] = 255;
            break;
        case DIRICHLET:
            cimg[y][x][0] = 78;
            cimg[y][x][1] = 78;
            cimg[y][x][2] = 78;
            break;
        case BODY:
            cimg[y][x][0] = 0;
            cimg[y][x][1] = 0;
            cimg[y][x][2] = 0;
            break;
        }
    }
}


__device__ F interpolate( F val, F y0, F x0, F y1, F x1 ) { return (val-x0)*(y1-y0)/(x1-x0) + y0; }
__device__ F base( F val )
{
    if ( val <= -0.75 ) return 0;
    else if ( val <= -0.25 ) return interpolate( val, 0.0, -0.75, 1.0, -0.25 );
    else if ( val <= 0.25 ) return 1.0;
    else if ( val <= 0.75 ) return interpolate( val, 1.0, 0.25, 0.0, 0.75 );
    else return 0.0;
}
__device__ F toM1P1(F in) { return in*2.0-1.0; }
__device__ F f_red( F gray ) { return base( toM1P1(gray) - 0.5 ); }
__device__ F f_green( F gray ) { return base( toM1P1(gray) ); }
__device__ F f_blue( F gray ) { return base( toM1P1(gray) + 0.5 ); }

__device__ void mapColor(F val, U8* color)
{
    if( val>1.0 ) val = 1.0;
    if( val<-1.0 ) val = -1.0;
    color[0] = (U8)(255.0*f_red(val) + 0.5);
    color[1] = (U8)(255.0*f_green(val) + 0.5);
    color[2] = (U8)(255.0*f_blue(val) + 0.5);
}

__global__ void generateImageTemp()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //Thread Index
    int tot = blockDim.x * gridDim.x; //Total Number of Threads
    for (int i=idx; i<TOTALNODES; i+=tot)
    {
        int y = i/WIDTH;
        int x = i-y*WIDTH;
        switch( cgrid[x][y].type )
        {
            case FREE:
            case DIRICHLET:
            {
                F t = (cgrid[x][y].T-ccolor_Tmin)/(ccolor_Tmax-ccolor_Tmin);
                mapColor(t,cimg[y][x]);
                break;
            }
            case BODY:
                cimg[y][x][0] = 0;
                cimg[y][x][1] = 0;
                cimg[y][x][2] = 0;
                break;
        }
    }
}

__global__ void generateImagePres()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //Thread Index
    int tot = blockDim.x * gridDim.x; //Total Number of Threads
    for (int i=idx; i<TOTALNODES; i+=tot)
    {
        int y = i/WIDTH;
        int x = i-y*WIDTH;
        switch( cgrid[x][y].type )
        {
            case FREE:
            case DIRICHLET:
            {
                F t = (cgrido[x][y].p-ccolor_pmin)/(ccolor_pmax-ccolor_pmin);
                mapColor(t,cimg[y][x]);
                break;
            }
            case BODY:
                cimg[y][x][0] = 0;
                cimg[y][x][1] = 0;
                cimg[y][x][2] = 0;
                break;
        }
    }
}

__global__ void generateImageRho()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //Thread Index
    int tot = blockDim.x * gridDim.x; //Total Number of Threads
    for (int i=idx; i<TOTALNODES; i+=tot)
    {
        int y = i/WIDTH;
        int x = i-y*WIDTH;
        switch( cgrid[x][y].type )
        {
            case FREE:
            case DIRICHLET:
            {
                F t = (cgrid[x][y].rho-ccolor_rhomin)/(ccolor_rhomax-ccolor_rhomin);
                mapColor(t,cimg[y][x]);
                break;
            }
            case BODY:
                cimg[y][x][0] = 0;
                cimg[y][x][1] = 0;
                cimg[y][x][2] = 0;
                break;
        }
    }
}

#define PUSHCONSTANT(X) CHECK( cudaMemcpyToSymbol(c##X,&X,sizeof(X),0, cudaMemcpyHostToDevice) )

void updateConstants()
{
    PUSHCONSTANT(color_Tmin);
    PUSHCONSTANT(color_Tmax);
    PUSHCONSTANT(color_pmin);
    PUSHCONSTANT(color_pmax);
    PUSHCONSTANT(color_rhomin);
    PUSHCONSTANT(color_rhomax);
}

/*************** CFD Computations ***************/


//differentiation stencils
#define FDx(def,var,x,y) cgrido##[x][y].d##var##dx+= (def[x+1][y].##var - def[x][y].##var  )*invPixelSize;
#define BDx(def,var,x,y) cgrido##[x][y].d##var##dx+= (def[x][y].##var   - def[x-1][y].##var)*invPixelSize;
#define FDy(def,var,x,y) cgrido##[x][y].d##var##dy+= (def[x][y+1].##var - def[x][y].##var  )*invPixelSize;
#define BDy(def,var,x,y) cgrido##[x][y].d##var##dy+= (def[x][y].##var   - def[x][y-1].##var)*invPixelSize;

#define FDx_local(def,var,x,y) d##var##dx+= (def[x+1][y].##var - def[x][y].##var   )*invPixelSize;
#define BDx_local(def,var,x,y) d##var##dx+= (def[x][y].##var   - def[x-1][y].##var )*invPixelSize;
#define FDy_local(def,var,x,y) d##var##dy+= (def[x][y+1].##var - def[x][y].##var   )*invPixelSize;
#define BDy_local(def,var,x,y) d##var##dy+= (def[x][y].##var   - def[x][y-1].##var )*invPixelSize;

#define inGridX(x)  (x>=0 && x<WIDTH)
#define inGridY(y)  (y>=0 && y<HEIGHT)
#define inGrid(x,y) (inGridX(x) && inGridY(y))

//upwind velocity tolerance
#define UPWIND_VTOL 1.0

__global__ void firstPass()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //Thread Index
    int tot = blockDim.x * gridDim.x; //Total Number of Threads
    for (int i=idx; i<TOTALNODES; i+=tot)
    {
        int y = i/WIDTH;
        int x = i-y*WIDTH;

        //Calculate Average Velocity
        F count = 1.0;
        cgrido[x][y].vmx = cgrid[x][y].vx;
        if( inGridX(x-1) )
        {
            cgrido[x][y].vmx += cgrid[x-1][y].vx;
            count++;
        }
        if( inGridX(x+1) )
        {
            cgrido[x][y].vmx += cgrid[x+1][y].vx;
            count++;
        }
        cgrido[x][y].vmx /= count;
        count = 1.0;
        cgrido[x][y].vmy = cgrid[x][y].vy;
        if( inGridY(y-1) )
        {
            cgrido[x][y].vmy += cgrid[x][y-1].vy;
            count++;
        }
        if( inGridY(y+1) )
        {
            cgrido[x][y].vmy += cgrid[x][y+1].vy;
            count++;
        }
        cgrido[x][y].vmy /= count;

        //Calculate Upwind Stencil
        bool UWXM = cgrido[x][y].vmx>=-UPWIND_VTOL || !inGridX(x+1) || cgrid[x+1][y].type == BODY || cgrid[x-1][y].type == DIRICHLET;
        bool UWXP = cgrido[x][y].vmx<= UPWIND_VTOL || !inGridX(x-1) || cgrid[x-1][y].type == BODY || cgrid[x+1][y].type == DIRICHLET;
        bool UWYM = cgrido[x][y].vmy>=-UPWIND_VTOL || !inGridX(y+1) || cgrid[x][y+1].type == BODY || cgrid[x][y-1].type == DIRICHLET;
        bool UWYP = cgrido[x][y].vmy<= UPWIND_VTOL || !inGridY(y-1) || cgrid[x][y-1].type == BODY || cgrid[x][y+1].type == DIRICHLET;

        if( cgrid[x][y].type == FREE || cgrid[x][y].type == DIRICHLET )
        {
            //cgrido[x][y].u = CV_AIR*cgrid[x][y].T;
            cgrido[x][y].p = cgrid[x][y].rho*R_AIR*cgrid[x][y].T;

            //Differentiate Velocity
            cgrido[x][y].dvxdx = 0.0;
            cgrido[x][y].dvxdy = 0.0;
            cgrido[x][y].dvydx = 0.0;
            cgrido[x][y].dvydy = 0.0;
            if( inGridX(x-1) && UWXM )
            {
                BDx(cgrid,vx,x,y);
                BDx(cgrid,vy,x,y);
            }
            if( inGridX(x+1) && UWXP )
            {
                FDx(cgrid,vx,x,y);
                FDx(cgrid,vy,x,y);
            }
            if( inGridY(y-1) && UWYM )
            {
                BDy(cgrid,vx,x,y);
                BDy(cgrid,vy,x,y);
            }
            if( inGridY(y+1) && UWYP )
            {
                FDy(cgrid,vx,x,y);
                FDy(cgrid,vy,x,y);
            }
            cgrido[x][y].dvxdx *= (1.0/3.0);
            cgrido[x][y].dvxdy *= (1.0/3.0);
            cgrido[x][y].dvydx *= (1.0/3.0);
            cgrido[x][y].dvydy *= (1.0/3.0);

            //Differentiate Temperature and Density
            cgrido[x][y].dTdx = 0.0;
            cgrido[x][y].dTdy = 0.0;
            cgrido[x][y].drhodx = 0.0;
            cgrido[x][y].drhody = 0.0;

            count = 0.0;
            if( inGridX(x-1) && UWXM && cgrid[x-1][y].type!=BODY )
            {
                count++;
                BDx(cgrid,T,x,y);
                BDx(cgrid,rho,x,y);
            }
            if( inGridX(x+1) && UWXP && cgrid[x+1][y].type!=BODY )
            {
                count++;
                FDx(cgrid,T,x,y);
                FDx(cgrid,rho,x,y);
            }
            if( count>0.0 )
            {
                cgrido[x][y].dTdx   /= count;
                cgrido[x][y].drhodx /= count;
            }
            count = 0.0;
            if( inGridY(y-1) && UWYM && cgrid[x][y-1].type!=BODY )
            {
                count++;
                BDy(cgrid,T,x,y);
                BDy(cgrid,rho,x,y);
            }
            if( inGridY(y+1) && UWYP && cgrid[x][y+1].type!=BODY )
            {
                count++;
                FDy(cgrid,T,x,y);
                FDy(cgrid,rho,x,y);
            }
            if( count>0.0 )
            {
                cgrido[x][y].dTdy   /= count;
                cgrido[x][y].drhody /= count;
            }

            //Compute The Cauchy Stress Tensor
            F mu = C1_AIR*cgrid[x][y].T*sqrt(cgrid[x][y].T)/(cgrid[x][y].T + S_AIR);
            //mu = 0.0;
            cgrido[x][y].sigmaxx = -cgrido[x][y].p + mu*(cgrido[x][y].dvxdx - cgrido[x][y].dvydy);
            cgrido[x][y].sigmayy = -cgrido[x][y].p + mu*(cgrido[x][y].dvydy - cgrido[x][y].dvxdx);
            cgrido[x][y].sigmaxy = mu*(cgrido[x][y].dvxdy + cgrido[x][y].dvydx);
            //cgrido[x][y].sigmaxx = -cgrido[x][y].p;
            //cgrido[x][y].sigmayy = -cgrido[x][y].p;
            //cgrido[x][y].sigmaxy = 0.0;
        }
    }
}

__global__ void secondPass()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //Thread Index
    int tot = blockDim.x * gridDim.x; //Total Number of Threads
    for (int i=idx; i<TOTALNODES; i+=tot)
    {
        int y = i/WIDTH;
        int x = i-y*WIDTH;
        if( cgrid[x][y].type == FREE )
        {
            F drhodt = -cgrid[x][y].rho*(cgrido[x][y].dvxdx + cgrido[x][y].dvydy)
                      - cgrido[x][y].drhodx*cgrido[x][y].vmx - cgrido[x][y].drhody*cgrido[x][y].vmy;
            F invrho = 1.0/cgrid[x][y].rho;
            assert(!isnan(cgrid[x][y].rho));

            //Calculate Upwind Stencil
            bool UWXM = cgrido[x][y].vmx>=-UPWIND_VTOL || !inGridX(x+1) || cgrid[x+1][y].type == BODY || cgrid[x-1][y].type == DIRICHLET;
            bool UWXP = cgrido[x][y].vmx<= UPWIND_VTOL || !inGridX(x-1) || cgrid[x-1][y].type == BODY || cgrid[x+1][y].type == DIRICHLET;
            bool UWYM = cgrido[x][y].vmy>=-UPWIND_VTOL || !inGridX(y+1) || cgrid[x][y+1].type == BODY || cgrid[x][y-1].type == DIRICHLET;
            bool UWYP = cgrido[x][y].vmy<= UPWIND_VTOL || !inGridY(y-1) || cgrid[x][y-1].type == BODY || cgrid[x][y+1].type == DIRICHLET;

            //Final set of gradients
            F dsigmaxxdx = 0.0;
            F dsigmaxydx = 0.0;
            F dsigmayydy = 0.0;
            F dsigmaxydy = 0.0;

            assert(!isnan(cgrido[x][y].sigmaxx));
            
            F count=0.0;
            if( inGridX(x-1) && UWXM && cgrid[x-1][y].type!=BODY )
            {
                count++;
                BDx_local(cgrido,sigmaxx,x,y);
                BDx_local(cgrido,sigmaxy,x,y);
            }
            if( inGridX(x+1) && UWXP && cgrid[x+1][y].type!=BODY )
            {
                count++;
                FDx_local(cgrido,sigmaxx,x,y);
                FDx_local(cgrido,sigmaxy,x,y);
            }
            if( count>0.0 )
            {
                dsigmaxxdx /= count;
                dsigmaxydx /= count;
            }
            count = 0.0;
            if( inGridY(y-1) && UWYM && cgrid[x][y-1].type!=BODY )
            {
                count++;
                BDy_local(cgrido,sigmayy,x,y);
                BDy_local(cgrido,sigmaxy,x,y);
            }
            if( inGridY(y+1) && UWYP && cgrid[x][y+1].type!=BODY )
            {
                count++;
                FDy_local(cgrido,sigmayy,x,y);
                FDy_local(cgrido,sigmaxy,x,y);
            }
            if( count>0.0 )
            {
                dsigmayydy /= count;
                dsigmaxydy /= count;
            }

            assert(!isnan(dsigmaxxdx));
            assert(!isnan(dsigmaxydy));
            assert(!isnan(cgrido[x][y].dvxdx));
            assert(!isnan(cgrido[x][y].dvxdy));


            F dvxdt = invrho*( dsigmaxxdx + dsigmaxydy )
                -cgrido[x][y].dvxdx*cgrido[x][y].vmx -cgrido[x][y].dvxdy*cgrido[x][y].vmy;
            if(isnan(dvxdt))
                printf("%.15E %.15E %.15E %.15E %.15E %.15E %.15E\n",
                    invrho,dsigmaxxdx,dsigmaxydy,
                    cgrido[x][y].dvxdx,cgrido[x][y].vmx,cgrido[x][y].dvxdy,cgrido[x][y].vmy);
            F dvydt = invrho*( dsigmayydy + dsigmaxydx )
                -cgrido[x][y].dvxdy*cgrido[x][y].vmx -cgrido[x][y].dvydy*cgrido[x][y].vmy;
            //F dudt = invrho*(
            //    cgrido[x][y].sigmaxx*cgrido[x][y].dvxdx +
            //    cgrido[x][y].sigmaxy*cgrido[x][y].dvydx +
            //    cgrido[x][y].sigmaxy*cgrido[x][y].dvxdy +
            //    cgrido[x][y].sigmayy*cgrido[x][y].dvydy
            //) - dudx*cgrid[x][y].vmx - dudy*cgrid[x][y].vmy;
            //F dTdt = dudt/CV_AIR;
            F dTdt = (invrho/CV_AIR)*(
                cgrido[x][y].sigmaxx*cgrido[x][y].dvxdx +
                cgrido[x][y].sigmaxy*cgrido[x][y].dvydx +
                cgrido[x][y].sigmaxy*cgrido[x][y].dvxdy +
                cgrido[x][y].sigmayy*cgrido[x][y].dvydy
            ) - cgrido[x][y].dTdx*cgrido[x][y].vmx - cgrido[x][y].dTdy*cgrido[x][y].vmy;


            //Propagate Everything!!
            assert(!isnan(drhodt));
            assert(!isnan(dvxdt));
            assert(!isnan(dvydt));
            //cgrid[x][y].rho = old_cgrid[x][y].rho + drhodt*cdt;
            cgrid[x][y].T   = old_cgrid[x][y].T   + dTdt*cdt;
            cgrid[x][y].vx  = old_cgrid[x][y].vx  + dvxdt*cdt;
            cgrid[x][y].vy  = old_cgrid[x][y].vy  + dvydt*cdt;
        }
    }
}

__global__ void makeCopy()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //Thread Index
    int tot = blockDim.x * gridDim.x; //Total Number of Threads
    for (int i=idx; i<TOTALNODES; i+=tot)
    {
        int y = i/WIDTH;
        int x = i-y*WIDTH;
        old_cgrid[x][y] = cgrid[x][y];
    }
}


void simStep()
{
    uploadGrid();
    updateConstants();
    makeCopy<<<CB,CT>>>();
    for( int i=0; i<2; i++ )
    {
        firstPass<<<CB,CT>>>();
        secondPass<<<CB,CT>>>();
    }
    downloadGrid();
    simPaint();
}

