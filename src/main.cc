/*
* main.cc
* 
* Copyright (c) 2017-2018 Pradu Kannan. All rights reserved.
*/

#include "nuklear.hh"
#include <GLFW/glfw3.h>
#include <stdio.h>

#include "defs.hh"
#include "simstate.hh"

struct nk_context* nkctx; //gui-state
//Test to see if a point is in the GUI
bool inGui(int x, int y)
{
    struct nk_window *iter;
    iter = nkctx->begin;
    while (iter)
    {
        if (x >= iter->bounds.x && x < iter->bounds.x + iter->bounds.w &&
            y >= iter->bounds.y && y < iter->bounds.y + iter->bounds.h)
            return true;
        struct nk_rect r = nk_window_get_bounds(nkctx);
        iter = iter->next;
    }
    return false;
}
void gui()
{

    /* init gui state */
    nk_glfw3_new_frame();

    /* Options Window */
    if (nk_begin(nkctx, "Options", nk_rect(WIDTH-250, 25, 230, 250),
        NK_WINDOW_BORDER|NK_WINDOW_MOVABLE|NK_WINDOW_SCALABLE|
        NK_WINDOW_MINIMIZABLE|NK_WINDOW_TITLE))
    {
        nk_layout_row_dynamic(nkctx, 30, 2);
        if(nk_button_label(nkctx, "Clear"))
            simClear();
        if (nk_button_label(nkctx, "Fill"))
            simFill();
        //nk_label(nkctx, "Pen Size:", NK_TEXT_LEFT);
        //cursorSize = nk_slide_int(nkctx, 0, cursorSize, 255, 1);
        nk_layout_row_dynamic(nkctx, 25, 1);
        nk_property_int(nkctx, "Pen  Size:", 0, &cursorSize, 100, 1, 1);

        //enum {EASY, HARD};
        //static int op = EASY;
        //nk_layout_row_static(nkctx, 30, 80, 1);
        //if (nk_button_label(nkctx, "button"))
        //    fprintf(stdout, "button pressed\n");

        //nk_layout_row_dynamic(nkctx, 30, 2);
        //if (nk_option_label(nkctx, "easy", op == EASY)) op = EASY;
        //if (nk_option_label(nkctx, "hard", op == HARD)) op = HARD;

        //nk_layout_row_dynamic(nkctx, 20, 1);
        //nk_label(nkctx, "background:", NK_TEXT_LEFT);
        //nk_layout_row_dynamic(nkctx, 25, 1);
        //if (nk_combo_begin_color(nkctx, background, nk_vec2(nk_widget_width(nkctx),400))) {
        //    nk_layout_row_dynamic(nkctx, 120, 1);
        //    background = nk_color_picker(nkctx, background, NK_RGBA);
        //    nk_layout_row_dynamic(nkctx, 25, 1);
        //    background.r = (nk_byte)nk_propertyi(nkctx, "#R:", 0, background.r, 255, 1,1);
        //    background.g = (nk_byte)nk_propertyi(nkctx, "#G:", 0, background.g, 255, 1,1);
        //    background.b = (nk_byte)nk_propertyi(nkctx, "#B:", 0, background.b, 255, 1,1);
        //    background.a = (nk_byte)nk_propertyi(nkctx, "#A:", 0, background.a, 255, 1,1);
        //    nk_combo_end(nkctx);
        //}
    }
    nk_end(nkctx);
    if (nk_begin(nkctx, "Drawing", nk_rect(25, 25, 230, 250),
        NK_WINDOW_BORDER|NK_WINDOW_MOVABLE|NK_WINDOW_SCALABLE|
        NK_WINDOW_MINIMIZABLE|NK_WINDOW_TITLE))
    {
        //nk_layout_row_dynamic(nkctx, 30, 1);
        //nk_label(nkctx, "Pen Size:", NK_TEXT_LEFT);
        //cursorSize = nk_slide_int(nkctx, 0, cursorSize, 255, 1);
        nk_layout_row_dynamic(nkctx, 25, 1);
        nk_property_int(nkctx, "Pen  Size:", 0, &cursorSize, 100, 1, 1);
    }
    nk_end(nkctx);
}

void error_callback(int error, const char* description)
{
    fprintf(stderr, "cfdbox: error : %s\n", description);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if( action==GLFW_PRESS ) switch(key)
    {
        case GLFW_KEY_F1:
        {
            static int toggle = 1;
            toggle = !toggle;
            nk_window_show(nkctx, "Drawing", (nk_show_states)toggle);
            break;
        }
        case GLFW_KEY_F2:
        {
            static int toggle = 1;
            toggle = !toggle;
            nk_window_show(nkctx, "Options", (nk_show_states)toggle);
            break;
        }
    }
    switch (key)
    {
    case GLFW_KEY_LEFT_SHIFT:
        if (action == GLFW_PRESS)
            LShift = true;
        else if (action == GLFW_RELEASE)
            LShift = false;
        break;
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    int xx = static_cast<int>(x + 0.5);
    int yy = static_cast<int>(y + 0.5);
    if(action == GLFW_PRESS && !inGui(xx,yy))
    {
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
        {
            Lmouse = true;
            simMouseMove(xx, yy);
        }
        if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
        {
            Rmouse = true;
            simMouseMove(xx, yy);
        }
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
        Lmouse = false;
    else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
        Rmouse = false;
    nk_glfw3_mouse_button_callback(window, button, action, mods);
}

static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos)
{
    int x = static_cast<int>(xpos + 0.5);
    int y = static_cast<int>(ypos + 0.5);
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

    //Setup GUI
    glEnable(GL_TEXTURE_2D);
    nkctx = nk_glfw3_init(window, NK_GLFW3_INSTALL_CALLBACKS);
    struct nk_font_atlas *atlas;
    nk_glfw3_font_stash_begin(&atlas);
    nk_glfw3_font_stash_end();
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    {
        struct nk_color table[NK_COLOR_COUNT];
        table[NK_COLOR_TEXT] = nk_rgba(210, 210, 210, 255);
        table[NK_COLOR_WINDOW] = nk_rgba(57, 67, 71, 215);
        table[NK_COLOR_HEADER] = nk_rgba(51, 51, 56, 220);
        table[NK_COLOR_BORDER] = nk_rgba(46, 46, 46, 255);
        table[NK_COLOR_BUTTON] = nk_rgba(48, 83, 111, 255);
        table[NK_COLOR_BUTTON_HOVER] = nk_rgba(58, 93, 121, 255);
        table[NK_COLOR_BUTTON_ACTIVE] = nk_rgba(63, 98, 126, 255);
        table[NK_COLOR_TOGGLE] = nk_rgba(50, 58, 61, 255);
        table[NK_COLOR_TOGGLE_HOVER] = nk_rgba(45, 53, 56, 255);
        table[NK_COLOR_TOGGLE_CURSOR] = nk_rgba(48, 83, 111, 255);
        table[NK_COLOR_SELECT] = nk_rgba(57, 67, 61, 255);
        table[NK_COLOR_SELECT_ACTIVE] = nk_rgba(48, 83, 111, 255);
        table[NK_COLOR_SLIDER] = nk_rgba(50, 58, 61, 255);
        table[NK_COLOR_SLIDER_CURSOR] = nk_rgba(48, 83, 111, 245);
        table[NK_COLOR_SLIDER_CURSOR_HOVER] = nk_rgba(53, 88, 116, 255);
        table[NK_COLOR_SLIDER_CURSOR_ACTIVE] = nk_rgba(58, 93, 121, 255);
        table[NK_COLOR_PROPERTY] = nk_rgba(50, 58, 61, 255);
        table[NK_COLOR_EDIT] = nk_rgba(50, 58, 61, 225);
        table[NK_COLOR_EDIT_CURSOR] = nk_rgba(210, 210, 210, 255);
        table[NK_COLOR_COMBO] = nk_rgba(50, 58, 61, 255);
        table[NK_COLOR_CHART] = nk_rgba(50, 58, 61, 255);
        table[NK_COLOR_CHART_COLOR] = nk_rgba(48, 83, 111, 255);
        table[NK_COLOR_CHART_COLOR_HIGHLIGHT] = nk_rgba(255, 0, 0, 255);
        table[NK_COLOR_SCROLLBAR] = nk_rgba(50, 58, 61, 255);
        table[NK_COLOR_SCROLLBAR_CURSOR] = nk_rgba(48, 83, 111, 255);
        table[NK_COLOR_SCROLLBAR_CURSOR_HOVER] = nk_rgba(53, 88, 116, 255);
        table[NK_COLOR_SCROLLBAR_CURSOR_ACTIVE] = nk_rgba(58, 93, 121, 255);
table[NK_COLOR_TAB_HEADER] = nk_rgba(48, 83, 111, 255);
        nk_style_from_table(nkctx, table);
    }
    

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
        gui();
        nk_glfw3_render(NK_ANTI_ALIASING_ON);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    simDestroy();
    return 0;
}

