/*
* defs.hh
* 
* Copyright (c) 2018 Pradu Kannan. All rights reserved.
*/

#ifndef ___DEFS_HH
#define ___DEFS_HH

#define WIDTH  1280
#define HEIGHT 720

#define TOTALNODES (WIDTH*HEIGHT)
#define CUDATHREADS 1024
#define CUDABLOCKS (TOTALNODES/CUDATHREADS)

#define CT CUDATHREADS
#define CB CUDABLOCKS

#define BOLTZMAN 1.38064852E-23
#define AVAGADRO 6.0221409E23
#define RBAR  (BOLTZMAN*AVAGADRO)

#define N_MOLAR (0.0140067)
#define O_MOLAR (0.015999)
#define H_MOLAR (0.00100794)
#define N2_MOLAR (2.0*N_MOLAR)
#define O2_MOLAR (2.0*O_MOLAR)
#define H2_MOLAR (2.0*H_MOLAR)
#define H20_MOLAR (O2_MOLAR + H_MOLAR)

#define N2_DOF  7
#define O2_DOF  7
#define H2_DOF  7
#define H2O_DOF 7

#define AIR_MOLAR 0.02897
#define R_AIR (RBAR/AIR_MOLAR)
#define GAMMA_AIR 1.4
#define CP_AIR ((GAMMA_AIR*R_AIR)/(GAMMA_AIR-1))
#define CV_AIR (CP_AIR/GAMMA_AIR)

//Parameters for Viscosity of Air: Sutherland's law
//mu = C1*T*sqrt(T)/(T+S)
#define C1_AIR 1.458E-6
#define S_AIR 110.4

enum{ FREE, DIRICHLET, BODY };

#endif  // ___DEFS_HH

