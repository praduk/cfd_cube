/*
* main.cc
* 
* Copyright (c) 2017 Pradu Kannan. All rights reserved.
*/

#include <stdio.h>
#include <GLFW/glfw3.h>


void error_callback(int error, const char* description)
{
    fprintf(stderr, "cfdbox: error : %s\n", description);
}

int main()
{
    if( !glfwInit() )
        return 1;

    glfwSetErrorCallback(error_callback);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
    GLFWwindow* window = glfwCreateWindow(640, 480, "CFDBox", NULL, NULL);
    if (!window)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    while (!glfwWindowShouldClose(window))
    {
        //Run Simulation
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

