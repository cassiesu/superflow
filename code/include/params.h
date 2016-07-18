#ifndef _TESTPARAMSH_
#define _TESTPARAMSH_

#include "image.h"
#include "opticalflow.h"
#include "io.h"

struct TestParams{
    image_t* wx;
    image_t* wy;
    
    color_image_t* im1;
    color_image_t* im2;
    
    image_t* match_x;
    image_t* match_y;
    image_t* match_z;
    
    optical_flow_params_t* params;
    
};

#endif