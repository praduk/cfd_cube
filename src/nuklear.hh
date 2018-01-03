/*
* nuklear.hh
* 
* Copyright (c) 2018 Pradu Kannan. All rights reserved.
*/

#ifndef __NUKLEAR_HH
#define __NUKLEAR_HH

#ifdef _WIN32
#include <windows.h>
# ifndef _CRT_SECURE_NO_WARNINGS
#  define _CRT_SECURE_NO_WARNINGS
# endif
#endif
#ifdef __APPLE__
# include <OpenGL/gl3.h>
# include <OpenGL/glu.h>
#else
# include <GL/gl.h>
# include <GL/glu.h>
#endif

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#include "nuklear.h"
#include "nuklear_glfw_gl2.h"
NK_API void nk_glfw3_mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

#endif