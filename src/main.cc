/*
* main.cc
* 
* Copyright (c) 2017-2018 Pradu Kannan. All rights reserved.
*/

#include <stdio.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include "defs.hh"
#include "simstate.hh"

void error_callback(int error, const char* description)
{
    fprintf(stderr, "cfdbox: error : %s\n", description);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    switch(key)
    {
        case GLFW_KEY_1: cursorSize=1;  break;
        case GLFW_KEY_2: cursorSize=2;  break;
        case GLFW_KEY_3: cursorSize=3;  break;
        case GLFW_KEY_4: cursorSize=4;  break;
        case GLFW_KEY_5: cursorSize=5;  break;
        case GLFW_KEY_6: cursorSize=6;  break;
        case GLFW_KEY_7: cursorSize=7;  break;
        case GLFW_KEY_8: cursorSize=8;  break;
        case GLFW_KEY_9: cursorSize=9;  break;
        case GLFW_KEY_0: cursorSize=0;  break;
        case GLFW_KEY_C: simClear();  break;
    }
    //if (key == GLFW_KEY_E && action == GLFW_PRESS)
    //    activate_airship();
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        Lmouse = true;
        simMouseMove(lastX,lastY);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
    {
        Rmouse = true;
        simMouseMove(lastX,lastY);
    }
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
        Lmouse = false;
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
        Rmouse = false;
}

static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos)
{
    int x = xpos + 0.5;
    int y = ypos + 0.5;
    simMouseMove(x,y);
}


int main()
{
    simInit();
    //Setup GLFW Window
    if( !glfwInit() )
        return 1;
    glfwSetErrorCallback(error_callback);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "CFDBox", NULL, NULL);
    if (!window)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    
    //Setup GL
    glClearColor(0.0f, 0.0f, 0.5f, 0.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, WIDTH, HEIGHT, 0);        
    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, WIDTH, HEIGHT);

    glTexImage2D(GL_TEXTURE_2D,0,3,WIDTH,HEIGHT,0,GL_RGB,GL_UNSIGNED_BYTE, image);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP); 

    glEnable(GL_TEXTURE_2D);

    //Run Simulation
    while (!glfwWindowShouldClose(window))
    {
        simStep();
        glClear(GL_COLOR_BUFFER_BIT);
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*)image);
        glBegin( GL_QUADS );
            glTexCoord2d(0.0, 0.0); glVertex2d(0.0,0.0);
            glTexCoord2d(1.0, 0.0); glVertex2d(WIDTH, 0.0);
            glTexCoord2d(1.0, 1.0); glVertex2d(WIDTH, HEIGHT);
            glTexCoord2d(0.0, 1.0); glVertex2d(0.0,HEIGHT);
        glEnd();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    simDestroy();
    return 0;
}

