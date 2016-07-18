#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <FreeImage.h>
#include <stdbool.h>
#include <x86intrin.h>
#include <string.h>
#include "image.h"
#include "solver.h"

#define BRK()  do { printf("%s %d\n", __FILE__, __LINE__); getchar(); } while (0)


//THIS IS A SLOW VERSION BUT READABLE
//Perform n iterations of the sor_coupled algorithm
//du and dv are used as initial guesses
//The system form is the same as in opticalflow.c
void sor_coupled_slow_but_readable(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, const int iterations, const float omega){
    int i,j,iter;
    float sigma_u,sigma_v,sum_dpsis,A11,A22,A12,B1,B2,det;
    for(iter = 0 ; iter<iterations ; iter++){
        for(j=0 ; j<du->height ; j++){
            for(i=0 ; i<du->width ; i++){
                sigma_u = 0.0f;
                sigma_v = 0.0f;
                sum_dpsis = 0.0f;

                // check left
                if(j>0){
                    sigma_u -= dpsis_vert->data[(j-1)*du->stride+i]*du->data[(j-1)*du->stride+i];
                    sigma_v -= dpsis_vert->data[(j-1)*du->stride+i]*dv->data[(j-1)*du->stride+i];
                    sum_dpsis += dpsis_vert->data[(j-1)*du->stride+i];
                }
                // check up
                if(i>0){
                    sigma_u -= dpsis_horiz->data[j*du->stride+i-1]*du->data[j*du->stride+i-1];
                    sigma_v -= dpsis_horiz->data[j*du->stride+i-1]*dv->data[j*du->stride+i-1];
                    sum_dpsis += dpsis_horiz->data[j*du->stride+i-1];
                }
                // check right
                if(j<du->height-1){
                    sigma_u -= dpsis_vert->data[j*du->stride+i]*du->data[(j+1)*du->stride+i];
                    sigma_v -= dpsis_vert->data[j*du->stride+i]*dv->data[(j+1)*du->stride+i];
                    sum_dpsis += dpsis_vert->data[j*du->stride+i];
                }
                // check down
                if(i<du->width-1){
                    sigma_u -= dpsis_horiz->data[j*du->stride+i]*du->data[j*du->stride+i+1];
                    sigma_v -= dpsis_horiz->data[j*du->stride+i]*dv->data[j*du->stride+i+1];
                    sum_dpsis += dpsis_horiz->data[j*du->stride+i];
                }
                A11 = a11->data[j*du->stride+i]+sum_dpsis;
                A12 = a12->data[j*du->stride+i];
                A22 = a22->data[j*du->stride+i]+sum_dpsis;
                det = A11*A22-A12*A12;
                B1 = b1->data[j*du->stride+i]-sigma_u;
                B2 = b2->data[j*du->stride+i]-sigma_v;
                du->data[j*du->stride+i] = (1.0f-omega)*du->data[j*du->stride+i] +omega*( A22*B1-A12*B2)/det;
                dv->data[j*du->stride+i] = (1.0f-omega)*dv->data[j*du->stride+i] +omega*(-A12*B1+A11*B2)/det;
            }
        }
    }
}

// This is a version that performs several precomputations.
// Operation count: 
void sor_coupled_slow_precompute_index(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, const int iterations, const float omega){
    int i,j,iter;
    float sigma_u,sigma_v,sum_dpsis,A11,A22,A12,B1,B2,det;

    // get the stride and index as saparate variables.
    int s;
    int jsi;
    int idx;
    float one_minus_omega;

    one_minus_omega = 1.0f - omega;
    s = du->stride;

    for(iter = 0 ; iter<iterations ; iter++){
        for(j=0 ; j<du->height ; j++){
            for(i=0 ; i<du->width ; i++){

                jsi = j*s+i;

                sigma_u = 0.0f;
                sigma_v = 0.0f;
                sum_dpsis = 0.0f;

                if(j>0){
                    idx = jsi - s;
                    sigma_u -= dpsis_vert->data[idx]*du->data[idx];
                    sigma_v -= dpsis_vert->data[idx]*dv->data[idx];
                    sum_dpsis += dpsis_vert->data[idx];
                }
                if(i>0){
                    idx = jsi - 1;
                    sigma_u -= dpsis_horiz->data[idx]*du->data[idx];
                    sigma_v -= dpsis_horiz->data[idx]*dv->data[idx];
                    sum_dpsis += dpsis_horiz->data[idx];
                }
                if(j<du->height-1){
                    idx = jsi + s;
                    sigma_u -= dpsis_vert->data[idx]*du->data[idx];
                    sigma_v -= dpsis_vert->data[idx]*dv->data[idx];
                    sum_dpsis += dpsis_vert->data[idx];
                }
                if(i<du->width-1){
                    idx = jsi + 1;
                    sigma_u -= dpsis_horiz->data[idx]*du->data[idx];
                    sigma_v -= dpsis_horiz->data[idx]*dv->data[idx];
                    sum_dpsis += dpsis_horiz->data[idx];
                }


                A11 = a11->data[jsi]+sum_dpsis;
                A12 = a12->data[jsi];
                A22 = a22->data[jsi]+sum_dpsis;
                det = A11*A22-A12*A12;
                B1 = b1->data[jsi]-sigma_u;
                B2 = b2->data[jsi]-sigma_v;

                // Forward substitution!
                du->data[jsi] = one_minus_omega*du->data[jsi] +omega*( A22*B1-A12*B2)/det;
                dv->data[jsi] = one_minus_omega*dv->data[jsi] +omega*(-A12*B1+A11*B2)/det;
            }
        }
    }
}

// This is a version that performs several precomputations.
// Operation count: 
void sor_coupled_slow_scalar_replacement(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, const int iterations, const float omega){
    int i,j,iter;
    float sigma_u,sigma_v,sum_dpsis,A11,A22,A12,B1,B2,det;

    // get the stride and index as saparate variables.
    int s;
    int jsi;
    int idx;
    float one_minus_omega;
    float dpsis_data;

    one_minus_omega = 1.0f - omega;
    s = du->stride;

    for(iter = 0 ; iter<iterations ; iter++){
        for(j=0 ; j<du->height ; j++){
            for(i=0 ; i<du->width ; i++){
                sigma_u = 0.0f;
                sigma_v = 0.0f;
                sum_dpsis = 0.0f;

                jsi = j*s+i;

                if(j>0){
                    idx = jsi - s;
                    dpsis_data = dpsis_vert->data[idx];

                    sigma_u -= dpsis_data*du->data[idx];
                    sigma_v -= dpsis_data*dv->data[idx];
                    sum_dpsis += dpsis_data;
                }
                if(i>0){
                    idx = jsi - 1;
                    dpsis_data = dpsis_horiz->data[idx];

                    sigma_u -= dpsis_data*du->data[idx];
                    sigma_v -= dpsis_data*dv->data[idx];
                    sum_dpsis += dpsis_data;
                }
                if(j<du->height-1){
                    idx = jsi + s;
                    dpsis_data = dpsis_vert->data[idx];

                    sigma_u -= dpsis_data*du->data[idx];
                    sigma_v -= dpsis_data*dv->data[idx];
                    sum_dpsis += dpsis_data;
                }
                if(i<du->width-1){
                    idx = jsi + 1;
                    dpsis_data = dpsis_horiz->data[idx];

                    sigma_u -= dpsis_data*du->data[idx];
                    sigma_v -= dpsis_data*dv->data[idx];
                    sum_dpsis += dpsis_data;
                }


                A11 = a11->data[jsi]+sum_dpsis;
                A12 = a12->data[jsi];
                A22 = a22->data[jsi]+sum_dpsis;
                det = A11*A22-A12*A12;
                B1 = b1->data[jsi]-sigma_u;
                B2 = b2->data[jsi]-sigma_v;

                // Forward substitution!
                du->data[jsi] = one_minus_omega*du->data[jsi] +omega*( A22*B1-A12*B2)/det;
                dv->data[jsi] = one_minus_omega*dv->data[jsi] +omega*(-A12*B1+A11*B2)/det;
            }
        }
    }
}

/* THIS IS A FASTER VERSION BUT UNREADABLE
the first iteration is done separately from the other to compute the inverse of the 2x2 block diagonal
the loop over pixels is split in different sections: the first line, the middle lines and the last line are split
for each line, the main for loop over columns is done 4 by 4, with the first and last one done independently
only work if width>=2 & height>=2 & iterations>=1*/
void sor_coupled(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, int iterations, float omega)
{
    if(du->width<2 || du->height<2 || iterations < 1)
        sor_coupled_slow_but_readable(du,dv,a11,a12,a22,b1,b2,dpsis_horiz,dpsis_vert,iterations, omega);

    int i,                                         // index for rows
            j,                                           // index columns
            iter,                                        // index of iteration
            incr_line = du->stride - du->width + 1,      // increment to pass from the last column to the first column at the next row
            ibefore,                                     // index of columns for the first part of the iteration
            nbefore = (du->width-2)%4;                   // number of columns to add to have a multiple of 4 (minus 2 for the first and last colum)
    int ifst = du->width-2-nbefore,                // first value of i when decreasing, do not count first and last column, and column to have a multiple of 4
            jfst = du->height-2;                         // first value of j when decreasing without counting first and last line

    // to avoid compute them many times
    int stride = du->stride;
    int stride1 = stride+1;
    int stride2 = stride1+1;
    int stride3 = stride2+1;
    int stride_ = -stride;
    int stride_1 = stride_+1;
    int stride_2 = stride_1+1;
    int stride_3 = stride_2+1;

    // [A11 A12 ; A12 A22] = inv([a11 a12 ; a12 a22]) including the dpsis component
    image_t *A11 = image_new(du->width,du->height),
            *A12 = image_new(du->width,du->height),
            *A22 = image_new(du->width,du->height);

    float sigma_u,sigma_v,       // contains the sum of the dpsis multiply by u or v coefficient in the line except the diagonal one
            sum_dpsis,                 // sum of the dpsis coefficient in a line
            det,                       // local variable to compute determinant
            B1,B2,                     // local variable

    // Next variables as used to move along the images
            *du_ptr = du->data, *dv_ptr = dv->data,
            *a11_ptr = a11->data, *a12_ptr = a12->data, *a22_ptr = a22->data,
            *b1_ptr = b1->data, *b2_ptr = b2->data,
            *dpsis_horiz_ptr = dpsis_horiz->data, *dpsis_vert_ptr = dpsis_vert->data,
            *A11_ptr = A11->data, *A12_ptr = A12->data, *A22_ptr = A22->data;

    // ---------------- FIRST ITERATION ----------------- //
    // reminder: inv([a b ; b c]) = [c -b ; -b a]/(ac-bb)
    //
    // for each pixel, compute sum_dpsis, sigma_u and sigma_v
    // add sum_dpsis to a11 and a22 and compute the determinant
    // deduce A11 A12 A22
    // compute B1 and B2
    // update du and dv
    // update pointer

    int count = 0;
    // ------------ first line, first column
    sum_dpsis = dpsis_horiz_ptr[0]           + dpsis_vert_ptr[0]                    ;
    sigma_u   = dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[0]*du_ptr[stride] ;
    sigma_v   = dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[0]*dv_ptr[stride] ;
    A22_ptr[0] = a11_ptr[0]+sum_dpsis;
    A11_ptr[0] = a22_ptr[0]+sum_dpsis;
    A12_ptr[0] = -a12_ptr[0];
    det = A11_ptr[0]*A22_ptr[0] - A12_ptr[0]*A12_ptr[0];
    A11_ptr[0] /= det;
    A22_ptr[0] /= det;
    A12_ptr[0] /= det;
    B1 = b1_ptr[0]+sigma_u;
    B2 = b2_ptr[0]+sigma_v;
    du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
    dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
    du_ptr++; dv_ptr++;
    a11_ptr++; a12_ptr++; a22_ptr++;
    A11_ptr++; A12_ptr++; A22_ptr++;
    b1_ptr++; b2_ptr++;
    dpsis_horiz_ptr++; dpsis_vert_ptr++;

    count++;
    // ------------ first line, column just after the first one to have a multiple of 4
    for(ibefore = nbefore ; ibefore-- ; ) // faster than for(ibefore = 0 ; ibefore < nbefore ; ibefore--)
    {
        sum_dpsis = dpsis_horiz_ptr[-1]            + dpsis_horiz_ptr[0]           + dpsis_vert_ptr[0]                    ;
        sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[0]*du_ptr[stride] ;
        sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[0]*dv_ptr[stride] ;
        A22_ptr[0] = a11_ptr[0]+sum_dpsis;
        A11_ptr[0] = a22_ptr[0]+sum_dpsis;
        A12_ptr[0] = -a12_ptr[0];
        det = A11_ptr[0]*A22_ptr[0] - A12_ptr[0]*A12_ptr[0];
        A11_ptr[0] /= det;
        A22_ptr[0] /= det;
        A12_ptr[0] /= det;
        B1 = b1_ptr[0]+sigma_u;
        B2 = b2_ptr[0]+sigma_v;
        du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
        dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
        du_ptr++; dv_ptr++;
        a11_ptr++; a12_ptr++; a22_ptr++;
        A11_ptr++; A12_ptr++; A22_ptr++;
        b1_ptr++; b2_ptr++;
        dpsis_horiz_ptr++; dpsis_vert_ptr++;
        count++;
    }

    // ------------ first line, other columns by 4
    for(i = ifst ; i ; i-=4)

    {
        // 1
        sum_dpsis = dpsis_horiz_ptr[-1]            + dpsis_horiz_ptr[0]           + dpsis_vert_ptr[0]                    ;
        sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[0]*du_ptr[stride] ;
        sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[0]*dv_ptr[stride] ;
        A22_ptr[0] = a11_ptr[0]+sum_dpsis;
        A11_ptr[0] = a22_ptr[0]+sum_dpsis;
        A12_ptr[0] = -a12_ptr[0];
        det = A11_ptr[0]*A22_ptr[0] - A12_ptr[0]*A12_ptr[0];
        A11_ptr[0] /= det;
        A22_ptr[0] /= det;
        A12_ptr[0] /= det;
        B1 = b1_ptr[0]+sigma_u;
        B2 = b2_ptr[0]+sigma_v;
        du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
        dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
        // 2
        sum_dpsis = dpsis_horiz_ptr[0]           + dpsis_horiz_ptr[1]           + dpsis_vert_ptr[1]                      ;
        sigma_u   = dpsis_horiz_ptr[0]*du_ptr[0] + dpsis_horiz_ptr[1]*du_ptr[2] + dpsis_vert_ptr[1]*du_ptr[stride1] ;
        sigma_v   = dpsis_horiz_ptr[0]*dv_ptr[0] + dpsis_horiz_ptr[1]*dv_ptr[2] + dpsis_vert_ptr[1]*dv_ptr[stride1] ;
        A22_ptr[1] = a11_ptr[1]+sum_dpsis;
        A11_ptr[1] = a22_ptr[1]+sum_dpsis;
        A12_ptr[1] = -a12_ptr[1];
        det = A11_ptr[1]*A22_ptr[1] - A12_ptr[1]*A12_ptr[1];
        A11_ptr[1] /= det;
        A22_ptr[1] /= det;
        A12_ptr[1] /= det;
        B1 = b1_ptr[1]+sigma_u;
        B2 = b2_ptr[1]+sigma_v;
        du_ptr[1] += omega*( A11_ptr[1]*B1 + A12_ptr[1]*B2 - du_ptr[1] );
        dv_ptr[1] += omega*( A12_ptr[1]*B1 + A22_ptr[1]*B2 - dv_ptr[1] );
        // 3
        sum_dpsis = dpsis_horiz_ptr[1]           + dpsis_horiz_ptr[2]           + dpsis_vert_ptr[2]                      ;
        sigma_u   = dpsis_horiz_ptr[1]*du_ptr[1] + dpsis_horiz_ptr[2]*du_ptr[3] + dpsis_vert_ptr[2]*du_ptr[stride2] ;
        sigma_v   = dpsis_horiz_ptr[1]*dv_ptr[1] + dpsis_horiz_ptr[2]*dv_ptr[3] + dpsis_vert_ptr[2]*dv_ptr[stride2] ;
        A22_ptr[2] = a11_ptr[2]+sum_dpsis;
        A11_ptr[2] = a22_ptr[2]+sum_dpsis;
        A12_ptr[2] = -a12_ptr[2];
        det = A11_ptr[2]*A22_ptr[2] - A12_ptr[2]*A12_ptr[2];
        A11_ptr[2] /= det;
        A22_ptr[2] /= det;
        A12_ptr[2] /= det;
        B1 = b1_ptr[2]+sigma_u;
        B2 = b2_ptr[2]+sigma_v;
        du_ptr[2] += omega*( A11_ptr[2]*B1 + A12_ptr[2]*B2 - du_ptr[2] );
        dv_ptr[2] += omega*( A12_ptr[2]*B1 + A22_ptr[2]*B2 - dv_ptr[2] );
        // 4
        sum_dpsis = dpsis_horiz_ptr[2]           + dpsis_horiz_ptr[3]           + dpsis_vert_ptr[3]                      ;
        sigma_u   = dpsis_horiz_ptr[2]*du_ptr[2] + dpsis_horiz_ptr[3]*du_ptr[4] + dpsis_vert_ptr[3]*du_ptr[stride3] ;
        sigma_v   = dpsis_horiz_ptr[2]*dv_ptr[2] + dpsis_horiz_ptr[3]*dv_ptr[4] + dpsis_vert_ptr[3]*dv_ptr[stride3] ;
        A22_ptr[3] = a11_ptr[3]+sum_dpsis;
        A11_ptr[3] = a22_ptr[3]+sum_dpsis;
        A12_ptr[3] = -a12_ptr[3];
        det = A11_ptr[3]*A22_ptr[3] - A12_ptr[3]*A12_ptr[3];
        A11_ptr[3] /= det;
        A22_ptr[3] /= det;
        A12_ptr[3] /= det;
        B1 = b1_ptr[3]+sigma_u;
        B2 = b2_ptr[3]+sigma_v;
        du_ptr[3] += omega*( A11_ptr[3]*B1 + A12_ptr[3]*B2 - du_ptr[3] );
        dv_ptr[3] += omega*( A12_ptr[3]*B1 + A22_ptr[3]*B2 - dv_ptr[3] );
        // increment pointer
        du_ptr += 4; dv_ptr += 4;
        a11_ptr += 4; a12_ptr += 4; a22_ptr += 4;
        A11_ptr += 4; A12_ptr += 4; A22_ptr += 4;
        b1_ptr += 4; b2_ptr += 4;
        dpsis_horiz_ptr += 4; dpsis_vert_ptr += 4;
        count+=4;
    }

    // ------------ first line, last column
    sum_dpsis = dpsis_horiz_ptr[-1]            + dpsis_vert_ptr[0]                    ;
    sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_vert_ptr[0]*du_ptr[stride] ;
    sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_vert_ptr[0]*dv_ptr[stride] ;
    A22_ptr[0] = a11_ptr[0]+sum_dpsis;
    A11_ptr[0] = a22_ptr[0]+sum_dpsis;
    A12_ptr[0] = -a12_ptr[0];
    det = A11_ptr[0]*A22_ptr[0] - A12_ptr[0]*A12_ptr[0];
    A11_ptr[0] /= det;
    A22_ptr[0] /= det;
    A12_ptr[0] /= det;
    B1 = b1_ptr[0]+sigma_u;
    B2 = b2_ptr[0]+sigma_v;
    du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
    dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
    // increment pointer to the next line
    du_ptr += incr_line; dv_ptr += incr_line;
    a11_ptr += incr_line; a12_ptr += incr_line; a22_ptr += incr_line;
    A11_ptr += incr_line; A12_ptr += incr_line; A22_ptr += incr_line;
    b1_ptr += incr_line; b2_ptr += incr_line;
    dpsis_horiz_ptr += incr_line; dpsis_vert_ptr += incr_line;
    count++;
    // ------------ line in the middle
    //for(j = jfst ; j-- ; )    // fast than for(j=1 ; j<du->height-1 ; j--)
    for(j=1 ; j<du->height-1 ; j++)
    {

        // ------------ line in the middle, first column
        sum_dpsis = dpsis_horiz_ptr[0]           + dpsis_vert_ptr[stride_]                     + dpsis_vert_ptr[0]                    ;
        sigma_u   = dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[stride_]*du_ptr[stride_] + dpsis_vert_ptr[0]*du_ptr[stride] ;
        sigma_v   = dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[stride_]*dv_ptr[stride_] + dpsis_vert_ptr[0]*dv_ptr[stride] ;
        A22_ptr[0] = a11_ptr[0]+sum_dpsis;
        A11_ptr[0] = a22_ptr[0]+sum_dpsis;
        A12_ptr[0] = -a12_ptr[0];
        det = A11_ptr[0]*A22_ptr[0] - A12_ptr[0]*A12_ptr[0];
        A11_ptr[0] /= det;
        A22_ptr[0] /= det;
        A12_ptr[0] /= det;
        B1 = b1_ptr[0]+sigma_u;
        B2 = b2_ptr[0]+sigma_v;
        du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
        dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
        du_ptr++; dv_ptr++;
        a11_ptr++; a12_ptr++; a22_ptr++;
        A11_ptr++; A12_ptr++; A22_ptr++;
        b1_ptr++; b2_ptr++;
        dpsis_horiz_ptr++; dpsis_vert_ptr++;
        count++;
        // ------------ line in the middle, column just after the first one to have a multiple of 4
        //for(ibefore = nbefore ; ibefore-- ; ) // faster than for(ibefore = 0 ; ibefore < nbefore ; ibefore--)
        for(ibefore = 0 ; ibefore < nbefore ; ibefore++)
        {
            sum_dpsis = dpsis_horiz_ptr[-1]            + dpsis_horiz_ptr[0]           + dpsis_vert_ptr[stride_]                     + dpsis_vert_ptr[0]                    ;
            sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[stride_]*du_ptr[stride_] + dpsis_vert_ptr[0]*du_ptr[stride] ;
            sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[stride_]*dv_ptr[stride_] + dpsis_vert_ptr[0]*dv_ptr[stride] ;
            A22_ptr[0] = a11_ptr[0]+sum_dpsis;
            A11_ptr[0] = a22_ptr[0]+sum_dpsis;
            A12_ptr[0] = -a12_ptr[0];
            det = A11_ptr[0]*A22_ptr[0] - A12_ptr[0]*A12_ptr[0];
            A11_ptr[0] /= det;
            A22_ptr[0] /= det;
            A12_ptr[0] /= det;
            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
            du_ptr++; dv_ptr++;
            a11_ptr++; a12_ptr++; a22_ptr++;
            A11_ptr++; A12_ptr++; A22_ptr++;
            b1_ptr++; b2_ptr++;
            dpsis_horiz_ptr++; dpsis_vert_ptr++;
            count++;
        }

        // ------------ line in the middle, other columns by 4
        for(i = 1; i <= ifst; i+=4)
        //for(i = ifst ; i ; i-=4)
        {
            // 1
            sum_dpsis = dpsis_horiz_ptr[-1]            + dpsis_horiz_ptr[0]           + dpsis_vert_ptr[stride_]                     + dpsis_vert_ptr[0]                    ;
            sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[stride_]*du_ptr[stride_] + dpsis_vert_ptr[0]*du_ptr[stride] ;
            sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[stride_]*dv_ptr[stride_] + dpsis_vert_ptr[0]*dv_ptr[stride] ;
            A22_ptr[0] = a11_ptr[0]+sum_dpsis;
            A11_ptr[0] = a22_ptr[0]+sum_dpsis;
            A12_ptr[0] = -a12_ptr[0];
            det = A11_ptr[0]*A22_ptr[0] - A12_ptr[0]*A12_ptr[0];
            A11_ptr[0] /= det;
            A22_ptr[0] /= det;
            A12_ptr[0] /= det;
            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
            // 2
            sum_dpsis = dpsis_horiz_ptr[0]           + dpsis_horiz_ptr[1]           + dpsis_vert_ptr[stride_1]                      + dpsis_vert_ptr[1]                      ;
            sigma_u   = dpsis_horiz_ptr[0]*du_ptr[0] + dpsis_horiz_ptr[1]*du_ptr[2] + dpsis_vert_ptr[stride_1]*du_ptr[stride_1] + dpsis_vert_ptr[1]*du_ptr[stride1] ;
            sigma_v   = dpsis_horiz_ptr[0]*dv_ptr[0] + dpsis_horiz_ptr[1]*dv_ptr[2] + dpsis_vert_ptr[stride_1]*dv_ptr[stride_1] + dpsis_vert_ptr[1]*dv_ptr[stride1] ;
            A22_ptr[1] = a11_ptr[1]+sum_dpsis;
            A11_ptr[1] = a22_ptr[1]+sum_dpsis;
            A12_ptr[1] = -a12_ptr[1];
            det = A11_ptr[1]*A22_ptr[1] - A12_ptr[1]*A12_ptr[1];
            A11_ptr[1] /= det;
            A22_ptr[1] /= det;
            A12_ptr[1] /= det;
            B1 = b1_ptr[1]+sigma_u;
            B2 = b2_ptr[1]+sigma_v;
            du_ptr[1] += omega*( A11_ptr[1]*B1 + A12_ptr[1]*B2 - du_ptr[1] );
            dv_ptr[1] += omega*( A12_ptr[1]*B1 + A22_ptr[1]*B2 - dv_ptr[1] );
            // 3
            sum_dpsis = dpsis_horiz_ptr[1]           + dpsis_horiz_ptr[2]           + dpsis_vert_ptr[stride_2]                      + dpsis_vert_ptr[2]                      ;
            sigma_u   = dpsis_horiz_ptr[1]*du_ptr[1] + dpsis_horiz_ptr[2]*du_ptr[3] + dpsis_vert_ptr[stride_2]*du_ptr[stride_2] + dpsis_vert_ptr[2]*du_ptr[stride2] ;
            sigma_v   = dpsis_horiz_ptr[1]*dv_ptr[1] + dpsis_horiz_ptr[2]*dv_ptr[3] + dpsis_vert_ptr[stride_2]*dv_ptr[stride_2] + dpsis_vert_ptr[2]*dv_ptr[stride2] ;
            A22_ptr[2] = a11_ptr[2]+sum_dpsis;
            A11_ptr[2] = a22_ptr[2]+sum_dpsis;
            A12_ptr[2] = -a12_ptr[2];
            det = A11_ptr[2]*A22_ptr[2] - A12_ptr[2]*A12_ptr[2];
            A11_ptr[2] /= det;
            A22_ptr[2] /= det;
            A12_ptr[2] /= det;
            B1 = b1_ptr[2]+sigma_u;
            B2 = b2_ptr[2]+sigma_v;
            du_ptr[2] += omega*( A11_ptr[2]*B1 + A12_ptr[2]*B2 - du_ptr[2] );
            dv_ptr[2] += omega*( A12_ptr[2]*B1 + A22_ptr[2]*B2 - dv_ptr[2] );
            // 4
            sum_dpsis = dpsis_horiz_ptr[2]           + dpsis_horiz_ptr[3]           + dpsis_vert_ptr[stride_3]                      + dpsis_vert_ptr[3]                      ;
            sigma_u   = dpsis_horiz_ptr[2]*du_ptr[2] + dpsis_horiz_ptr[3]*du_ptr[4] + dpsis_vert_ptr[stride_3]*du_ptr[stride_3] + dpsis_vert_ptr[3]*du_ptr[stride3] ;
            sigma_v   = dpsis_horiz_ptr[2]*dv_ptr[2] + dpsis_horiz_ptr[3]*dv_ptr[4] + dpsis_vert_ptr[stride_3]*dv_ptr[stride_3] + dpsis_vert_ptr[3]*dv_ptr[stride3] ;
            A22_ptr[3] = a11_ptr[3]+sum_dpsis;
            A11_ptr[3] = a22_ptr[3]+sum_dpsis;
            A12_ptr[3] = -a12_ptr[3];
            det = A11_ptr[3]*A22_ptr[3] - A12_ptr[3]*A12_ptr[3];
            A11_ptr[3] /= det;
            A22_ptr[3] /= det;
            A12_ptr[3] /= det;
            B1 = b1_ptr[3]+sigma_u;
            B2 = b2_ptr[3]+sigma_v;
            du_ptr[3] += omega*( A11_ptr[3]*B1 + A12_ptr[3]*B2 - du_ptr[3] );
            dv_ptr[3] += omega*( A12_ptr[3]*B1 + A22_ptr[3]*B2 - dv_ptr[3] );
            // increment pointer
            du_ptr += 4; dv_ptr += 4;
            a11_ptr += 4; a12_ptr += 4; a22_ptr += 4;
            A11_ptr += 4; A12_ptr += 4; A22_ptr += 4;
            b1_ptr += 4; b2_ptr += 4;
            dpsis_horiz_ptr += 4; dpsis_vert_ptr += 4;
            count+=4;
        }

        // ------------ line in the middle, last column
        sum_dpsis = dpsis_horiz_ptr[-1]            + dpsis_vert_ptr[stride_]                     + dpsis_vert_ptr[0]                    ;
        sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_vert_ptr[stride_]*du_ptr[stride_] + dpsis_vert_ptr[0]*du_ptr[stride] ;
        sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_vert_ptr[stride_]*dv_ptr[stride_] + dpsis_vert_ptr[0]*dv_ptr[stride] ;
        A22_ptr[0] = a11_ptr[0]+sum_dpsis;
        A11_ptr[0] = a22_ptr[0]+sum_dpsis;
        A12_ptr[0] = -a12_ptr[0];
        det = A11_ptr[0]*A22_ptr[0] - A12_ptr[0]*A12_ptr[0];
        A11_ptr[0] /= det;
        A22_ptr[0] /= det;
        A12_ptr[0] /= det;
        B1 = b1_ptr[0]+sigma_u;
        B2 = b2_ptr[0]+sigma_v;
        du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
        dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
        // increment pointer to the next line
        du_ptr += incr_line; dv_ptr += incr_line;
        a11_ptr += incr_line; a12_ptr += incr_line; a22_ptr += incr_line;
        A11_ptr += incr_line; A12_ptr += incr_line; A22_ptr += incr_line;
        b1_ptr += incr_line; b2_ptr += incr_line;
        dpsis_horiz_ptr += incr_line; dpsis_vert_ptr += incr_line;
        count++;
    }

    // ------------ last line, first column
    sum_dpsis = dpsis_horiz_ptr[0]           + dpsis_vert_ptr[stride_]                     ;
    sigma_u   = dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[stride_]*du_ptr[stride_] ;
    sigma_v   = dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[stride_]*dv_ptr[stride_] ;
    A22_ptr[0] = a11_ptr[0]+sum_dpsis;
    A11_ptr[0] = a22_ptr[0]+sum_dpsis;
    A12_ptr[0] = -a12_ptr[0];
    det = A11_ptr[0]*A22_ptr[0] - A12_ptr[0]*A12_ptr[0];
    A11_ptr[0] /= det;
    A22_ptr[0] /= det;
    A12_ptr[0] /= det;
    B1 = b1_ptr[0]+sigma_u;
    B2 = b2_ptr[0]+sigma_v;
    du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
    dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
    du_ptr++; dv_ptr++;
    a11_ptr++; a12_ptr++; a22_ptr++;
    A11_ptr++; A12_ptr++; A22_ptr++;
    b1_ptr++; b2_ptr++;
    dpsis_horiz_ptr++; dpsis_vert_ptr++;
    count++;

    // ------------ last line, column just after the first one to have a multiple of 4
    //for(ibefore = nbefore ; ibefore-- ; ) // faster than for(ibefore = 0 ; ibefore < nbefore ; ibefore--)
    for(ibefore = 0 ; ibefore < nbefore ; ibefore++)
    {
        sum_dpsis = dpsis_horiz_ptr[-1]            + dpsis_horiz_ptr[0]           + dpsis_vert_ptr[stride_]                     ;
        sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[stride_]*du_ptr[stride_] ;
        sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[stride_]*dv_ptr[stride_] ;
        A22_ptr[0] = a11_ptr[0]+sum_dpsis;
        A11_ptr[0] = a22_ptr[0]+sum_dpsis;
        A12_ptr[0] = -a12_ptr[0];
        det = A11_ptr[0]*A22_ptr[0] - A12_ptr[0]*A12_ptr[0];
        A11_ptr[0] /= det;
        A22_ptr[0] /= det;
        A12_ptr[0] /= det;
        B1 = b1_ptr[0]+sigma_u;
        B2 = b2_ptr[0]+sigma_v;
        du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
        dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
        du_ptr++; dv_ptr++;
        a11_ptr++; a12_ptr++; a22_ptr++;
        A11_ptr++; A12_ptr++; A22_ptr++;
        b1_ptr++; b2_ptr++;
        dpsis_horiz_ptr++; dpsis_vert_ptr++;
        count++;
    }

    // ------------ last line, other columns by 4
    for(i = ifst ; i ; i-=4)
    //for(i = 1 ; i<=ifst ; i+=4)
    {
        // 1
        sum_dpsis = dpsis_horiz_ptr[-1]            + dpsis_horiz_ptr[0]           + dpsis_vert_ptr[stride_]                     ;
        sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[stride_]*du_ptr[stride_] ;
        sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[stride_]*dv_ptr[stride_] ;
        A22_ptr[0] = a11_ptr[0]+sum_dpsis;
        A11_ptr[0] = a22_ptr[0]+sum_dpsis;
        A12_ptr[0] = -a12_ptr[0];
        det = A11_ptr[0]*A22_ptr[0] - A12_ptr[0]*A12_ptr[0];
        A11_ptr[0] /= det;
        A22_ptr[0] /= det;
        A12_ptr[0] /= det;
        B1 = b1_ptr[0]+sigma_u;
        B2 = b2_ptr[0]+sigma_v;
        du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
        dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
        // 2
        sum_dpsis = dpsis_horiz_ptr[0]           + dpsis_horiz_ptr[1]           + dpsis_vert_ptr[stride_1]                      ;
        sigma_u   = dpsis_horiz_ptr[0]*du_ptr[0] + dpsis_horiz_ptr[1]*du_ptr[2] + dpsis_vert_ptr[stride_1]*du_ptr[stride_1] ;
        sigma_v   = dpsis_horiz_ptr[0]*dv_ptr[0] + dpsis_horiz_ptr[1]*dv_ptr[2] + dpsis_vert_ptr[stride_1]*dv_ptr[stride_1] ;
        A22_ptr[1] = a11_ptr[1]+sum_dpsis;
        A11_ptr[1] = a22_ptr[1]+sum_dpsis;
        A12_ptr[1] = -a12_ptr[1];
        det = A11_ptr[1]*A22_ptr[1] - A12_ptr[1]*A12_ptr[1];
        A11_ptr[1] /= det;
        A22_ptr[1] /= det;
        A12_ptr[1] /= det;
        B1 = b1_ptr[1]+sigma_u;
        B2 = b2_ptr[1]+sigma_v;
        du_ptr[1] += omega*( A11_ptr[1]*B1 + A12_ptr[1]*B2 - du_ptr[1] );
        dv_ptr[1] += omega*( A12_ptr[1]*B1 + A22_ptr[1]*B2 - dv_ptr[1] );
        // 3
        sum_dpsis = dpsis_horiz_ptr[1]           + dpsis_horiz_ptr[2]           + dpsis_vert_ptr[stride_2]                      ;
        sigma_u   = dpsis_horiz_ptr[1]*du_ptr[1] + dpsis_horiz_ptr[2]*du_ptr[3] + dpsis_vert_ptr[stride_2]*du_ptr[stride_2] ;
        sigma_v   = dpsis_horiz_ptr[1]*dv_ptr[1] + dpsis_horiz_ptr[2]*dv_ptr[3] + dpsis_vert_ptr[stride_2]*dv_ptr[stride_2] ;
        A22_ptr[2] = a11_ptr[2]+sum_dpsis;
        A11_ptr[2] = a22_ptr[2]+sum_dpsis;
        A12_ptr[2] = -a12_ptr[2];
        det = A11_ptr[2]*A22_ptr[2] - A12_ptr[2]*A12_ptr[2];
        A11_ptr[2] /= det;
        A22_ptr[2] /= det;
        A12_ptr[2] /= det;
        B1 = b1_ptr[2]+sigma_u;
        B2 = b2_ptr[2]+sigma_v;
        du_ptr[2] += omega*( A11_ptr[2]*B1 + A12_ptr[2]*B2 - du_ptr[2] );
        dv_ptr[2] += omega*( A12_ptr[2]*B1 + A22_ptr[2]*B2 - dv_ptr[2] );
        // 4
        sum_dpsis = dpsis_horiz_ptr[2]           + dpsis_horiz_ptr[3]           + dpsis_vert_ptr[stride_3]                      ;
        sigma_u   = dpsis_horiz_ptr[2]*du_ptr[2] + dpsis_horiz_ptr[3]*du_ptr[4] + dpsis_vert_ptr[stride_3]*du_ptr[stride_3] ;
        sigma_v   = dpsis_horiz_ptr[2]*dv_ptr[2] + dpsis_horiz_ptr[3]*dv_ptr[4] + dpsis_vert_ptr[stride_3]*dv_ptr[stride_3] ;
        A22_ptr[3] = a11_ptr[3]+sum_dpsis;
        A11_ptr[3] = a22_ptr[3]+sum_dpsis;
        A12_ptr[3] = -a12_ptr[3];
        det = A11_ptr[3]*A22_ptr[3] - A12_ptr[3]*A12_ptr[3];
        A11_ptr[3] /= det;
        A22_ptr[3] /= det;
        A12_ptr[3] /= det;
        B1 = b1_ptr[3]+sigma_u;
        B2 = b2_ptr[3]+sigma_v;
        du_ptr[3] += omega*( A11_ptr[3]*B1 + A12_ptr[3]*B2 - du_ptr[3] );
        dv_ptr[3] += omega*( A12_ptr[3]*B1 + A22_ptr[3]*B2 - dv_ptr[3] );
        // increment pointer
        du_ptr += 4; dv_ptr += 4;
        a11_ptr += 4; a12_ptr += 4; a22_ptr += 4;
        A11_ptr += 4; A12_ptr += 4; A22_ptr += 4;
        b1_ptr += 4; b2_ptr += 4;
        dpsis_horiz_ptr += 4; dpsis_vert_ptr += 4;
        count+=4;
    }

    // ------------ last line, last column
    sum_dpsis = dpsis_horiz_ptr[-1]            + dpsis_vert_ptr[stride_]                     ;
    sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_vert_ptr[stride_]*du_ptr[stride_] ;
    sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_vert_ptr[stride_]*dv_ptr[stride_] ;
    A22_ptr[0] = a11_ptr[0]+sum_dpsis;
    A11_ptr[0] = a22_ptr[0]+sum_dpsis;
    A12_ptr[0] = -a12_ptr[0];
    det = A11_ptr[0]*A22_ptr[0] - A12_ptr[0]*A12_ptr[0];
    A11_ptr[0] /= det;
    A22_ptr[0] /= det;
    A12_ptr[0] /= det;
    B1 = b1_ptr[0]+sigma_u;
    B2 = b2_ptr[0]+sigma_v;
    du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
    dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
    count++;

    if(du->height*du->width != count)
    {
        printf("sor_coupled. should be %d loops, count %d loops.", du->width*du->height, count);
        BRK();
    }


    // useless to increment here

    // ---------------- OTHER ITERATION ----------------- //
    // for each pixel, compute sigma_u and sigma_v
    // compute B1 and B2
    // update du and dv
    // update pointer

    for(iter = iterations ; --iter ; ) // faster than for(iter = 1 ; iter<iterations ; iter++)
    //for(iter = 1 ; iter<iterations ; iter++)
    {

        // set pointer to the beginning
        du_ptr = du->data; dv_ptr = dv->data;
        A11_ptr = A11->data; A12_ptr = A12->data; A22_ptr = A22->data;
        b1_ptr = b1->data; b2_ptr = b2->data;
        dpsis_horiz_ptr = dpsis_horiz->data; dpsis_vert_ptr = dpsis_vert->data;

        // process all elements as before

        // ------------ first line, first column
        sigma_u   = dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[0]*du_ptr[stride] ;
        sigma_v   = dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[0]*dv_ptr[stride] ;
        B1 = b1_ptr[0]+sigma_u;
        B2 = b2_ptr[0]+sigma_v;
        du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
        dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
        du_ptr++; dv_ptr++;
        A11_ptr++; A12_ptr++; A22_ptr++;
        b1_ptr++; b2_ptr++;
        dpsis_horiz_ptr++; dpsis_vert_ptr++;

        // ------------ first line, column just after the first one to have a multiple of 4
        for(ibefore = nbefore ; ibefore-- ; ) // faster than for(ibefore = 0 ; ibefore < nbefore ; ibefore--)
        //for(ibefore = 0 ; ibefore < nbefore ; ibefore++)
        {
            sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[0]*du_ptr[stride] ;
            sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[0]*dv_ptr[stride] ;
            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
            du_ptr++; dv_ptr++;
            A11_ptr++; A12_ptr++; A22_ptr++;
            b1_ptr++; b2_ptr++;
            dpsis_horiz_ptr++; dpsis_vert_ptr++;
        }

        // ------------ first line, other columns by 4
        for(i = ifst ; i ; i-=4)
        //for(i = 1 ; i<=ifst ; i+=4)
        {
            // 1
            sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[0]*du_ptr[stride] ;
            sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[0]*dv_ptr[stride] ;
            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
            // 2
            sigma_u   = dpsis_horiz_ptr[0]*du_ptr[0] + dpsis_horiz_ptr[1]*du_ptr[2] + dpsis_vert_ptr[1]*du_ptr[stride1] ;
            sigma_v   = dpsis_horiz_ptr[0]*dv_ptr[0] + dpsis_horiz_ptr[1]*dv_ptr[2] + dpsis_vert_ptr[1]*dv_ptr[stride1] ;
            B1 = b1_ptr[1]+sigma_u;
            B2 = b2_ptr[1]+sigma_v;
            du_ptr[1] += omega*( A11_ptr[1]*B1 + A12_ptr[1]*B2 - du_ptr[1] );
            dv_ptr[1] += omega*( A12_ptr[1]*B1 + A22_ptr[1]*B2 - dv_ptr[1] );
            // 3
            sigma_u   = dpsis_horiz_ptr[1]*du_ptr[1] + dpsis_horiz_ptr[2]*du_ptr[3] + dpsis_vert_ptr[2]*du_ptr[stride2] ;
            sigma_v   = dpsis_horiz_ptr[1]*dv_ptr[1] + dpsis_horiz_ptr[2]*dv_ptr[3] + dpsis_vert_ptr[2]*dv_ptr[stride2] ;
            B1 = b1_ptr[2]+sigma_u;
            B2 = b2_ptr[2]+sigma_v;
            du_ptr[2] += omega*( A11_ptr[2]*B1 + A12_ptr[2]*B2 - du_ptr[2] );
            dv_ptr[2] += omega*( A12_ptr[2]*B1 + A22_ptr[2]*B2 - dv_ptr[2] );
            // 4
            sigma_u   = dpsis_horiz_ptr[2]*du_ptr[2] + dpsis_horiz_ptr[3]*du_ptr[4] + dpsis_vert_ptr[3]*du_ptr[stride3] ;
            sigma_v   = dpsis_horiz_ptr[2]*dv_ptr[2] + dpsis_horiz_ptr[3]*dv_ptr[4] + dpsis_vert_ptr[3]*dv_ptr[stride3] ;
            B1 = b1_ptr[3]+sigma_u;
            B2 = b2_ptr[3]+sigma_v;
            du_ptr[3] += omega*( A11_ptr[3]*B1 + A12_ptr[3]*B2 - du_ptr[3] );
            dv_ptr[3] += omega*( A12_ptr[3]*B1 + A22_ptr[3]*B2 - dv_ptr[3] );
            // increment pointer
            du_ptr += 4; dv_ptr += 4;
            A11_ptr += 4; A12_ptr += 4; A22_ptr += 4;
            b1_ptr += 4; b2_ptr += 4;
            dpsis_horiz_ptr += 4; dpsis_vert_ptr += 4;
        }

        // ------------ first line, last column
        sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_vert_ptr[0]*du_ptr[stride] ;
        sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_vert_ptr[0]*dv_ptr[stride] ;
        B1 = b1_ptr[0]+sigma_u;
        B2 = b2_ptr[0]+sigma_v;
        du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
        dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
        // increment pointer to the next line
        du_ptr += incr_line; dv_ptr += incr_line;
        A11_ptr += incr_line; A12_ptr += incr_line; A22_ptr += incr_line;
        b1_ptr += incr_line; b2_ptr += incr_line;
        dpsis_horiz_ptr += incr_line; dpsis_vert_ptr += incr_line;

        // ------------ line in the middle
        for(j = jfst ; j-- ; )    // fast than for(j=1 ; j<du->height-1 ; j--)
        {

            // ------------ line in the middle, first column
            sigma_u   = dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[stride_]*du_ptr[stride_] + dpsis_vert_ptr[0]*du_ptr[stride] ;
            sigma_v   = dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[stride_]*dv_ptr[stride_] + dpsis_vert_ptr[0]*dv_ptr[stride] ;
            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
            du_ptr++; dv_ptr++;
            A11_ptr++; A12_ptr++; A22_ptr++;
            b1_ptr++; b2_ptr++;
            dpsis_horiz_ptr++; dpsis_vert_ptr++;

            // ------------ line in the middle, column just after the first one to have a multiple of 4
            for(ibefore = nbefore ; ibefore-- ; ) // faster than for(ibefore = 0 ; ibefore < nbefore ; ibefore--)
            {
                sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[stride_]*du_ptr[stride_] + dpsis_vert_ptr[0]*du_ptr[stride] ;
                sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[stride_]*dv_ptr[stride_] + dpsis_vert_ptr[0]*dv_ptr[stride] ;
                B1 = b1_ptr[0]+sigma_u;
                B2 = b2_ptr[0]+sigma_v;
                du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
                dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
                du_ptr++; dv_ptr++;
                A11_ptr++; A12_ptr++; A22_ptr++;
                b1_ptr++; b2_ptr++;
                dpsis_horiz_ptr++; dpsis_vert_ptr++;
            }

            // ------------ line in the middle, other columns by 4
            for(i = ifst ; i ; i-=4)
            {
                // 1
                sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[stride_]*du_ptr[stride_] + dpsis_vert_ptr[0]*du_ptr[stride] ;
                sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[stride_]*dv_ptr[stride_] + dpsis_vert_ptr[0]*dv_ptr[stride] ;
                B1 = b1_ptr[0]+sigma_u;
                B2 = b2_ptr[0]+sigma_v;
                du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
                dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
                // 2
                sigma_u   = dpsis_horiz_ptr[0]*du_ptr[0] + dpsis_horiz_ptr[1]*du_ptr[2] + dpsis_vert_ptr[stride_1]*du_ptr[stride_1] + dpsis_vert_ptr[1]*du_ptr[stride1] ;
                sigma_v   = dpsis_horiz_ptr[0]*dv_ptr[0] + dpsis_horiz_ptr[1]*dv_ptr[2] + dpsis_vert_ptr[stride_1]*dv_ptr[stride_1] + dpsis_vert_ptr[1]*dv_ptr[stride1] ;
                B1 = b1_ptr[1]+sigma_u;
                B2 = b2_ptr[1]+sigma_v;
                du_ptr[1] += omega*( A11_ptr[1]*B1 + A12_ptr[1]*B2 - du_ptr[1] );
                dv_ptr[1] += omega*( A12_ptr[1]*B1 + A22_ptr[1]*B2 - dv_ptr[1] );
                // 3
                sigma_u   = dpsis_horiz_ptr[1]*du_ptr[1] + dpsis_horiz_ptr[2]*du_ptr[3] + dpsis_vert_ptr[stride_2]*du_ptr[stride_2] + dpsis_vert_ptr[2]*du_ptr[stride2] ;
                sigma_v   = dpsis_horiz_ptr[1]*dv_ptr[1] + dpsis_horiz_ptr[2]*dv_ptr[3] + dpsis_vert_ptr[stride_2]*dv_ptr[stride_2] + dpsis_vert_ptr[2]*dv_ptr[stride2] ;
                B1 = b1_ptr[2]+sigma_u;
                B2 = b2_ptr[2]+sigma_v;
                du_ptr[2] += omega*( A11_ptr[2]*B1 + A12_ptr[2]*B2 - du_ptr[2] );
                dv_ptr[2] += omega*( A12_ptr[2]*B1 + A22_ptr[2]*B2 - dv_ptr[2] );
                // 4
                sigma_u   = dpsis_horiz_ptr[2]*du_ptr[2] + dpsis_horiz_ptr[3]*du_ptr[4] + dpsis_vert_ptr[stride_3]*du_ptr[stride_3] + dpsis_vert_ptr[3]*du_ptr[stride3] ;
                sigma_v   = dpsis_horiz_ptr[2]*dv_ptr[2] + dpsis_horiz_ptr[3]*dv_ptr[4] + dpsis_vert_ptr[stride_3]*dv_ptr[stride_3] + dpsis_vert_ptr[3]*dv_ptr[stride3] ;
                B1 = b1_ptr[3]+sigma_u;
                B2 = b2_ptr[3]+sigma_v;
                du_ptr[3] += omega*( A11_ptr[3]*B1 + A12_ptr[3]*B2 - du_ptr[3] );
                dv_ptr[3] += omega*( A12_ptr[3]*B1 + A22_ptr[3]*B2 - dv_ptr[3] );
                // increment pointer
                du_ptr += 4; dv_ptr += 4;
                A11_ptr += 4; A12_ptr += 4; A22_ptr += 4;
                b1_ptr += 4; b2_ptr += 4;
                dpsis_horiz_ptr += 4; dpsis_vert_ptr += 4;
            }

            // ------------ line in the middle, last column
            sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_vert_ptr[stride_]*du_ptr[stride_] + dpsis_vert_ptr[0]*du_ptr[stride] ;
            sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_vert_ptr[stride_]*dv_ptr[stride_] + dpsis_vert_ptr[0]*dv_ptr[stride] ;
            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
            // increment pointer to the next line
            du_ptr += incr_line; dv_ptr += incr_line;
            A11_ptr += incr_line; A12_ptr += incr_line; A22_ptr += incr_line;
            b1_ptr += incr_line; b2_ptr += incr_line;
            dpsis_horiz_ptr += incr_line; dpsis_vert_ptr += incr_line;

        }

        // ------------ last line, first column
        sigma_u   = dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[stride_]*du_ptr[stride_] ;
        sigma_v   = dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[stride_]*dv_ptr[stride_] ;
        B1 = b1_ptr[0]+sigma_u;
        B2 = b2_ptr[0]+sigma_v;
        du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
        dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
        du_ptr++; dv_ptr++;
        A11_ptr++; A12_ptr++; A22_ptr++;
        b1_ptr++; b2_ptr++;
        dpsis_horiz_ptr++; dpsis_vert_ptr++;

        // ------------ last line, column just after the first one to have a multiple of 4
        for(ibefore = nbefore ; ibefore-- ; ) // faster than for(ibefore = 0 ; ibefore < nbefore ; ibefore--)
        {
            sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[stride_]*du_ptr[stride_] ;
            sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[stride_]*dv_ptr[stride_] ;
            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
            du_ptr++; dv_ptr++;
            A11_ptr++; A12_ptr++; A22_ptr++;
            b1_ptr++; b2_ptr++;
            dpsis_horiz_ptr++; dpsis_vert_ptr++;
        }

        // ------------ last line, other columns by 4
        for(i = ifst ; i ; i-=4)
        {
            // 1
            sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1] + dpsis_vert_ptr[stride_]*du_ptr[stride_] ;
            sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1] + dpsis_vert_ptr[stride_]*dv_ptr[stride_] ;
            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
            // 2
            sigma_u   = dpsis_horiz_ptr[0]*du_ptr[0] + dpsis_horiz_ptr[1]*du_ptr[2] + dpsis_vert_ptr[stride_1]*du_ptr[stride_1] ;
            sigma_v   = dpsis_horiz_ptr[0]*dv_ptr[0] + dpsis_horiz_ptr[1]*dv_ptr[2] + dpsis_vert_ptr[stride_1]*dv_ptr[stride_1] ;
            B1 = b1_ptr[1]+sigma_u;
            B2 = b2_ptr[1]+sigma_v;
            du_ptr[1] += omega*( A11_ptr[1]*B1 + A12_ptr[1]*B2 - du_ptr[1] );
            dv_ptr[1] += omega*( A12_ptr[1]*B1 + A22_ptr[1]*B2 - dv_ptr[1] );
            // 3
            sigma_u   = dpsis_horiz_ptr[1]*du_ptr[1] + dpsis_horiz_ptr[2]*du_ptr[3] + dpsis_vert_ptr[stride_2]*du_ptr[stride_2] ;
            sigma_v   = dpsis_horiz_ptr[1]*dv_ptr[1] + dpsis_horiz_ptr[2]*dv_ptr[3] + dpsis_vert_ptr[stride_2]*dv_ptr[stride_2] ;
            B1 = b1_ptr[2]+sigma_u;
            B2 = b2_ptr[2]+sigma_v;
            du_ptr[2] += omega*( A11_ptr[2]*B1 + A12_ptr[2]*B2 - du_ptr[2] );
            dv_ptr[2] += omega*( A12_ptr[2]*B1 + A22_ptr[2]*B2 - dv_ptr[2] );
            // 4
            sigma_u   = dpsis_horiz_ptr[2]*du_ptr[2] + dpsis_horiz_ptr[3]*du_ptr[4] + dpsis_vert_ptr[stride_3]*du_ptr[stride_3] ;
            sigma_v   = dpsis_horiz_ptr[2]*dv_ptr[2] + dpsis_horiz_ptr[3]*dv_ptr[4] + dpsis_vert_ptr[stride_3]*dv_ptr[stride_3] ;
            B1 = b1_ptr[3]+sigma_u;
            B2 = b2_ptr[3]+sigma_v;
            du_ptr[3] += omega*( A11_ptr[3]*B1 + A12_ptr[3]*B2 - du_ptr[3] );
            dv_ptr[3] += omega*( A12_ptr[3]*B1 + A22_ptr[3]*B2 - dv_ptr[3] );
            // increment pointer
            du_ptr += 4; dv_ptr += 4;
            A11_ptr += 4; A12_ptr += 4; A22_ptr += 4;
            b1_ptr += 4; b2_ptr += 4;
            dpsis_horiz_ptr += 4; dpsis_vert_ptr += 4;
        }

        // ------------ last line, last column
        sigma_u   = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_vert_ptr[stride_]*du_ptr[stride_] ;
        sigma_v   = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_vert_ptr[stride_]*dv_ptr[stride_] ;
        B1 = b1_ptr[0]+sigma_u;
        B2 = b2_ptr[0]+sigma_v;
        du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
        dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
        // useless to increment here
    }

    // delete allocated images
    image_delete(A11); image_delete(A12); image_delete(A22);
}

void sor_coupled_blocked_1x4(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, int iterations, float omega)
{
    // Fall back to standard solver in case of trivial cases
    if(du->width < 2 || du->height < 2 || iterations < 1)
    {
        sor_coupled_slow_but_readable(du,dv,a11,a12,a22,b1,b2,dpsis_horiz,dpsis_vert,iterations, omega);
        return;
    }

    int i, j, iter;
    int stride = du->stride;
    int stride_ = -stride;
    int stride_1 = stride_ + 1;
    int stride_2 = stride_ + 2;
    int stride_3 = stride_ + 3;

    float *du_ptr = du->data, *dv_ptr = dv->data;
    float *a11_ptr = a11->data, *a12_ptr = a12->data, *a22_ptr = a22->data;
    float *b1_ptr = b1->data, *b2_ptr = b2->data;
    float *dpsis_horiz_ptr = dpsis_horiz->data, *dpsis_vert_ptr = dpsis_vert->data;

    // Typical exchage memory for speed
    image_t *A11 = image_new(du->width,du->height);
    image_t *A22 = image_new(du->width,du->height);
    image_t *A12 = image_new(du->width,du->height);

    float *A11_ptr = A11->data, *A12_ptr = A12->data, *A22_ptr = A22->data;


    float sum_dpsis, sigma_u, sigma_v, B1, B2, det;
    float sum_dpsis_1, sum_dpsis_2, sum_dpsis_3, sum_dpsis_4;

    // Should be able to hold 3 blocks simultaneously.
    int bsize = 4;
    // int npad_w = (du->width - 2) % bsize;
    int niter_w = (du->width - 2) / bsize;

    int i_bound = bsize * niter_w + 1;

    int new_line_incr = du->stride - du->width + 1;

    float A11_1, A11_2, A11_3, A11_4, A11_;
    float A12_1, A12_2, A12_3, A12_4, A12_;
    float A22_1, A22_2, A22_3, A22_4, A22_;

    // Use first iteration to calculate the determinant of matrix A
    // No blocking procedure for this round
    // Loop of 4 -> so that we have some degree of vectorization!
    // TODO: See if we can use inline function to simplify things a little bit

    // int count = 0;

    for (j = 0; j < du->height; ++j) {

        // Left cell
        sum_dpsis = dpsis_horiz_ptr[0];

        if(j > 0)
            sum_dpsis += dpsis_vert_ptr[stride_];
        if (j < du->height - 1)
            sum_dpsis += dpsis_vert_ptr[0];

        A22_ = a11_ptr[0]+sum_dpsis;
        A11_ = a22_ptr[0]+sum_dpsis;
        A12_ = -a12_ptr[0];

        det = A11_*A22_ - A12_*A12_;

        A11_ptr[0] = A11_ / det;
        A22_ptr[0] = A22_ / det;
        A12_ptr[0] = A12_ / det;

        dpsis_horiz_ptr++; dpsis_vert_ptr++;
        A11_ptr++; A12_ptr++; A22_ptr++;
        a11_ptr++; a12_ptr++; a22_ptr++;

        // count++;

        // first row
        for (i = 1;  i < i_bound; i+=4) {
            // Column 1
            sum_dpsis_1 = dpsis_horiz_ptr[-1] + dpsis_horiz_ptr[0];

            if(j > 0)
                sum_dpsis_1 += dpsis_vert_ptr[stride_];
            if (j < du->height - 1)
                sum_dpsis_1 += dpsis_vert_ptr[0];

            A22_1 = a11_ptr[0]+sum_dpsis_1;
            A11_1 = a22_ptr[0]+sum_dpsis_1;
            A12_1 = -a12_ptr[0];

            det = A11_1*A22_1 - A12_1*A12_1;

            A11_ptr[0] = A11_1 / det;
            A22_ptr[0] = A22_1 / det;
            A12_ptr[0] = A12_1 / det;

            // Column 2
            sum_dpsis_2 = dpsis_horiz_ptr[0] + dpsis_horiz_ptr[1];

            if(j > 0)
                sum_dpsis_2 += dpsis_vert_ptr[stride_1];
            if (j < du->height - 1)
                sum_dpsis_2 += dpsis_vert_ptr[1];

            A22_2 = a11_ptr[1]+sum_dpsis_2;
            A11_2 = a22_ptr[1]+sum_dpsis_2;
            A12_2 = -a12_ptr[1];

            det = A11_2*A22_2 - A12_2*A12_2;

            A11_ptr[1] = A11_2 / det;
            A22_ptr[1] = A22_2 / det;
            A12_ptr[1] = A12_2 / det;

            // Column 3
            sum_dpsis_3 = dpsis_horiz_ptr[1] + dpsis_horiz_ptr[2];

            if(j > 0)
                sum_dpsis_3 += dpsis_vert_ptr[stride_2];
            if (j < du->height - 1)
                sum_dpsis_3 += dpsis_vert_ptr[2];

            A22_3 = a11_ptr[2]+sum_dpsis_3;
            A11_3 = a22_ptr[2]+sum_dpsis_3;
            A12_3 = -a12_ptr[2];

            det = A11_3*A22_3 - A12_3*A12_3;

            A11_ptr[2] = A11_3 / det;
            A22_ptr[2] = A22_3 / det;
            A12_ptr[2] = A12_3 / det;

            // Column 4
            sum_dpsis_4 = dpsis_horiz_ptr[2] + dpsis_horiz_ptr[3];

            if(j > 0)
                sum_dpsis_4 += dpsis_vert_ptr[stride_3];
            if (j < du->height - 1)
                sum_dpsis_4 += dpsis_vert_ptr[3];

            A22_4 = a11_ptr[3]+sum_dpsis_4;
            A11_4 = a22_ptr[3]+sum_dpsis_4;
            A12_4 = -a12_ptr[3];

            det = A11_4*A22_4 - A12_4*A12_4;

            A11_ptr[3] = A11_4 / det;
            A22_ptr[3] = A22_4 / det;
            A12_ptr[3] = A12_4 / det;

            dpsis_horiz_ptr+=4; dpsis_vert_ptr+=4;
            A11_ptr+=4; A12_ptr+=4; A22_ptr+=4;
            a11_ptr+=4; a12_ptr+=4; a22_ptr+=4;
            //count += 4;
        }

        // Cope with whatever that's left
        while(i < du->width - 1)
        {
            sum_dpsis = dpsis_horiz_ptr[-1] + dpsis_horiz_ptr[0];

            if(j > 0)
                sum_dpsis += dpsis_vert_ptr[stride_];
            if (j < du->height - 1)
                sum_dpsis += dpsis_vert_ptr[0];

            A22_ = a11_ptr[0]+sum_dpsis;
            A11_ = a22_ptr[0]+sum_dpsis;
            A12_ = -a12_ptr[0];

            det = A11_*A22_ - A12_*A12_;

            A11_ptr[0] = A11_ / det;
            A22_ptr[0] = A22_ / det;
            A12_ptr[0] = A12_ / det;

            dpsis_horiz_ptr++; dpsis_vert_ptr++;
            A11_ptr++; A12_ptr++; A22_ptr++;
            a11_ptr++; a12_ptr++; a22_ptr++;

            i++;
            //count++;
        }

        // upperright corner
        sum_dpsis = dpsis_horiz_ptr[-1];

        if(j > 0)
            sum_dpsis += dpsis_vert_ptr[stride_];
        if (j < du->height - 1)
            sum_dpsis += dpsis_vert_ptr[0];

        A22_ = a11_ptr[0]+sum_dpsis;
        A11_ = a22_ptr[0]+sum_dpsis;
        A12_ = -a12_ptr[0];

        det = A11_*A22_ - A12_*A12_;

        A11_ptr[0] = A11_ / det;
        A22_ptr[0] = A22_ / det;
        A12_ptr[0] = A12_ / det;

        dpsis_horiz_ptr += new_line_incr;
        dpsis_vert_ptr += new_line_incr;
        A11_ptr+=new_line_incr; A12_ptr+=new_line_incr; A22_ptr+=new_line_incr;
        a11_ptr+=new_line_incr; a12_ptr+=new_line_incr; a22_ptr+=new_line_incr;
        //count++;
    }

//    if(du->height*du->width != count)
//    {
//        printf("height: %d, width: %d, iterations: %d\n", du->height, du->width, iterations);
//        printf("should be %d loops, count %d loops.", du->width*du->height, count);
//        BRK();
//    }

    //
    for (iter = 0; iter < iterations; ++iter) {

        int count = 0;

        // set pointer to the beginning
        du_ptr = du->data; dv_ptr = dv->data;
        A11_ptr = A11->data; A12_ptr = A12->data; A22_ptr = A22->data;
        b1_ptr = b1->data; b2_ptr = b2->data;
        dpsis_horiz_ptr = dpsis_horiz->data; dpsis_vert_ptr = dpsis_vert->data;


        for (j = 0; j < du->height; ++j) {
            // Left column
            sigma_u = dpsis_horiz_ptr[0]*du_ptr[1];
            sigma_v = dpsis_horiz_ptr[0]*dv_ptr[1];

            if(j > 0){
                sigma_u += dpsis_vert_ptr[stride_]*du_ptr[stride_];
                sigma_v += dpsis_vert_ptr[stride_]*dv_ptr[stride_];
            }

            if(j < du->height - 1){
                sigma_u += dpsis_vert_ptr[0]*du_ptr[stride];
                sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride];
            }

            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );
            du_ptr++; dv_ptr++;
            A11_ptr++; A12_ptr++; A22_ptr++;
            b1_ptr++; b2_ptr++;
            dpsis_horiz_ptr++; dpsis_vert_ptr++;
            count++;

            // There is no point in unrolling here. As each item depends exactly on its horizontal neighbors
            // ILP is impossible
            for(i = 1; i < du->width - 1; i++){
                sigma_u = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1];
                sigma_v = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1];

                if(j > 0){
                    sigma_u += dpsis_vert_ptr[stride_]*du_ptr[stride_];
                    sigma_v += dpsis_vert_ptr[stride_]*dv_ptr[stride_];
                }

                if(j < du->height - 1){
                    sigma_u += dpsis_vert_ptr[0]*du_ptr[stride];
                    sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride];
                }

                B1 = b1_ptr[0]+sigma_u;
                B2 = b2_ptr[0]+sigma_v;
                du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
                dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

                du_ptr++; dv_ptr++;
                A11_ptr++; A12_ptr++; A22_ptr++;
                b1_ptr++; b2_ptr++;
                dpsis_horiz_ptr++; dpsis_vert_ptr++;
                count++;
            }

            // right column
            sigma_u = dpsis_horiz_ptr[-1]*du_ptr[-1];
            sigma_v = dpsis_horiz_ptr[-1]*dv_ptr[-1];
            if(j > 0){
                sigma_u += dpsis_vert_ptr[stride_]*du_ptr[stride_];
                sigma_v += dpsis_vert_ptr[stride_]*dv_ptr[stride_];
            }

            if(j < du->height - 1){
                sigma_u += dpsis_vert_ptr[0]*du_ptr[stride];
                sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride];
            }

            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

            // increment pointer
            du_ptr += new_line_incr; dv_ptr += new_line_incr;
            A11_ptr += new_line_incr; A12_ptr += new_line_incr; A22_ptr += new_line_incr;
            b1_ptr += new_line_incr; b2_ptr += new_line_incr;
            dpsis_horiz_ptr += new_line_incr; dpsis_vert_ptr += new_line_incr;
            count++;
        }

        if(du->height*du->width != count)
        {
            printf("should be %d loops, count %d loops.", du->width*du->height, count);
            BRK();
        }
    }

    // delete allocated images
    image_delete(A11); image_delete(A12); image_delete(A22);

}

void sor_coupled_blocked_2x2(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, int iterations, float omega)
{
    // Fall back to standard solver in case of trivial cases
    if(du->width < 2 || du->height < 2 || iterations < 1)
    {
        sor_coupled_slow_but_readable(du,dv,a11,a12,a22,b1,b2,dpsis_horiz,dpsis_vert,iterations, omega);
        return;
    }

    int i, j, iter;
    int stride = du->stride;
    int stride_ = -stride;
    int stride_1 = stride_ + 1;
    int stride_2 = stride_ + 2;
    int stride_3 = stride_ + 3;


    float *du_ptr = du->data, *dv_ptr = dv->data;
    float *a11_ptr = a11->data, *a12_ptr = a12->data, *a22_ptr = a22->data;
    float *b1_ptr = b1->data, *b2_ptr = b2->data;
    float *dpsis_horiz_ptr = dpsis_horiz->data, *dpsis_vert_ptr = dpsis_vert->data;

    // Typical exchage memory for speed
    image_t *A11 = image_new(du->width,du->height);
    image_t *A22 = image_new(du->width,du->height);
    image_t *A12 = image_new(du->width,du->height);

    float *A11_ptr = A11->data, *A12_ptr = A12->data, *A22_ptr = A22->data;


    float sum_dpsis, sigma_u, sigma_v, B1, B2, det;
    float sum_dpsis_1, sum_dpsis_2, sum_dpsis_3, sum_dpsis_4;

    // Should be able to hold 3 blocks simultaneously.
    int bsize = 4;
    int niter_w = (du->width - 2) / bsize;
    int i_bound = bsize * niter_w + 1;

    int new_line_incr = du->stride - du->width + 1;

    float A11_1, A11_2, A11_3, A11_4, A11_;
    float A12_1, A12_2, A12_3, A12_4, A12_;
    float A22_1, A22_2, A22_3, A22_4, A22_;

    for (j = 0; j < du->height; ++j) {

        // Left cell
        sum_dpsis = dpsis_horiz_ptr[0];

        if(j > 0)
            sum_dpsis += dpsis_vert_ptr[stride_];
        if (j < du->height - 1)
            sum_dpsis += dpsis_vert_ptr[0];

        A22_ = a11_ptr[0]+sum_dpsis;
        A11_ = a22_ptr[0]+sum_dpsis;
        A12_ = -a12_ptr[0];

        det = A11_*A22_ - A12_*A12_;

        A11_ptr[0] = A11_ / det;
        A22_ptr[0] = A22_ / det;
        A12_ptr[0] = A12_ / det;

        dpsis_horiz_ptr++; dpsis_vert_ptr++;
        A11_ptr++; A12_ptr++; A22_ptr++;
        a11_ptr++; a12_ptr++; a22_ptr++;

        // first row
        for (i = 1;  i < i_bound; i+=4) {
            // Column 1
            sum_dpsis_1 = dpsis_horiz_ptr[-1] + dpsis_horiz_ptr[0];

            if(j > 0)
                sum_dpsis_1 += dpsis_vert_ptr[stride_];
            if (j < du->height - 1)
                sum_dpsis_1 += dpsis_vert_ptr[0];

            A22_1 = a11_ptr[0]+sum_dpsis_1;
            A11_1 = a22_ptr[0]+sum_dpsis_1;
            A12_1 = -a12_ptr[0];

            det = A11_1*A22_1 - A12_1*A12_1;

            A11_ptr[0] = A11_1 / det;
            A22_ptr[0] = A22_1 / det;
            A12_ptr[0] = A12_1 / det;

            // Column 2
            sum_dpsis_2 = dpsis_horiz_ptr[0] + dpsis_horiz_ptr[1];

            if(j > 0)
                sum_dpsis_2 += dpsis_vert_ptr[stride_1];
            if (j < du->height - 1)
                sum_dpsis_2 += dpsis_vert_ptr[1];

            A22_2 = a11_ptr[1]+sum_dpsis_2;
            A11_2 = a22_ptr[1]+sum_dpsis_2;
            A12_2 = -a12_ptr[1];

            det = A11_2*A22_2 - A12_2*A12_2;

            A11_ptr[1] = A11_2 / det;
            A22_ptr[1] = A22_2 / det;
            A12_ptr[1] = A12_2 / det;

            // Column 3
            sum_dpsis_3 = dpsis_horiz_ptr[1] + dpsis_horiz_ptr[2];

            if(j > 0)
                sum_dpsis_3 += dpsis_vert_ptr[stride_2];
            if (j < du->height - 1)
                sum_dpsis_3 += dpsis_vert_ptr[2];

            A22_3 = a11_ptr[2]+sum_dpsis_3;
            A11_3 = a22_ptr[2]+sum_dpsis_3;
            A12_3 = -a12_ptr[2];

            det = A11_3*A22_3 - A12_3*A12_3;

            A11_ptr[2] = A11_3 / det;
            A22_ptr[2] = A22_3 / det;
            A12_ptr[2] = A12_3 / det;

            // Column 4
            sum_dpsis_4 = dpsis_horiz_ptr[2] + dpsis_horiz_ptr[3];

            if(j > 0)
                sum_dpsis_4 += dpsis_vert_ptr[stride_3];
            if (j < du->height - 1)
                sum_dpsis_4 += dpsis_vert_ptr[3];

            A22_4 = a11_ptr[3]+sum_dpsis_4;
            A11_4 = a22_ptr[3]+sum_dpsis_4;
            A12_4 = -a12_ptr[3];

            det = A11_4*A22_4 - A12_4*A12_4;

            A11_ptr[3] = A11_4 / det;
            A22_ptr[3] = A22_4 / det;
            A12_ptr[3] = A12_4 / det;

            dpsis_horiz_ptr+=4; dpsis_vert_ptr+=4;
            A11_ptr+=4; A12_ptr+=4; A22_ptr+=4;
            a11_ptr+=4; a12_ptr+=4; a22_ptr+=4;
        }

        // Cope with whatever that's left
        while(i < du->width - 1)
        {
            sum_dpsis = dpsis_horiz_ptr[-1] + dpsis_horiz_ptr[0];

            if(j > 0)
                sum_dpsis += dpsis_vert_ptr[stride_];
            if (j < du->height - 1)
                sum_dpsis += dpsis_vert_ptr[0];

            A22_ = a11_ptr[0]+sum_dpsis;
            A11_ = a22_ptr[0]+sum_dpsis;
            A12_ = -a12_ptr[0];

            det = A11_*A22_ - A12_*A12_;

            A11_ptr[0] = A11_ / det;
            A22_ptr[0] = A22_ / det;
            A12_ptr[0] = A12_ / det;

            dpsis_horiz_ptr++; dpsis_vert_ptr++;
            A11_ptr++; A12_ptr++; A22_ptr++;
            a11_ptr++; a12_ptr++; a22_ptr++;
            i++;
        }

        // upperright corner
        sum_dpsis = dpsis_horiz_ptr[-1];

        if(j > 0)
            sum_dpsis += dpsis_vert_ptr[stride_];
        if (j < du->height - 1)
            sum_dpsis += dpsis_vert_ptr[0];

        A22_ = a11_ptr[0]+sum_dpsis;
        A11_ = a22_ptr[0]+sum_dpsis;
        A12_ = -a12_ptr[0];

        det = A11_*A22_ - A12_*A12_;

        A11_ptr[0] = A11_ / det;
        A22_ptr[0] = A22_ / det;
        A12_ptr[0] = A12_ / det;

        dpsis_horiz_ptr += new_line_incr;
        dpsis_vert_ptr += new_line_incr;
        A11_ptr+=new_line_incr; A12_ptr+=new_line_incr; A22_ptr+=new_line_incr;
        a11_ptr+=new_line_incr; a12_ptr+=new_line_incr; a22_ptr+=new_line_incr;
    }

    // Main iterations
    // Loop over 2x2 blocks. Special cases: left/right column, top/bottom row, possible one more extra row/column.
    int block_line_incr = du->stride + (du->stride - du->width + 1);
    int j_block_iter = (du->height - 2) / 2;
    int i_block_iter = (du->width - 2) / 2;
    bool odd_row = du->height % 2? true:false;
    bool odd_col = du->width % 2? true:false;

    for(iter = 0; iter < iterations; ++iter)
    {

        //int count = 0;
        // set pointer to the beginning
        du_ptr = du->data; dv_ptr = dv->data;
        A11_ptr = A11->data; A12_ptr = A12->data; A22_ptr = A22->data;
        b1_ptr = b1->data; b2_ptr = b2->data;
        dpsis_horiz_ptr = dpsis_horiz->data; dpsis_vert_ptr = dpsis_vert->data;

        // upperleft corner
        sigma_u = dpsis_horiz_ptr[0]*du_ptr[1];
        sigma_v = dpsis_horiz_ptr[0]*dv_ptr[1];
        sigma_u += dpsis_vert_ptr[0]*du_ptr[stride];
        sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride];
        B1 = b1_ptr[0]+sigma_u;
        B2 = b2_ptr[0]+sigma_v;
        du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
        dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

        du_ptr++; dv_ptr++;
        A11_ptr++; A12_ptr++; A22_ptr++;
        b1_ptr++; b2_ptr++;
        dpsis_horiz_ptr++; dpsis_vert_ptr++;
        // count++;
        // middle of the first line
        for( i = 1; i < du->width - 1; i++)
        {
            sigma_u = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1];
            sigma_v = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1];
            sigma_u += dpsis_vert_ptr[0]*du_ptr[stride];
            sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride];

            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

            du_ptr++; dv_ptr++;
            A11_ptr++; A12_ptr++; A22_ptr++;
            b1_ptr++; b2_ptr++;
            dpsis_horiz_ptr++; dpsis_vert_ptr++;
            //count++;
        }

        // rightupper corner
        sigma_u = dpsis_horiz_ptr[-1]*du_ptr[-1];
        sigma_v = dpsis_horiz_ptr[-1]*dv_ptr[-1];
        sigma_u += dpsis_vert_ptr[0]*du_ptr[stride];
        sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride];
        B1 = b1_ptr[0]+sigma_u;
        B2 = b2_ptr[0]+sigma_v;
        du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
        dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

        //send the pointer to next line
        du_ptr+=new_line_incr; dv_ptr+=new_line_incr;
        b1_ptr+=new_line_incr; b2_ptr+=new_line_incr;
        dpsis_horiz_ptr += new_line_incr; dpsis_vert_ptr += new_line_incr;
        A11_ptr+=new_line_incr; A12_ptr+=new_line_incr; A22_ptr+=new_line_incr;

        //count++;
        for(j = 0; j < j_block_iter; ++j)
        {

            // First column
            sigma_u = dpsis_horiz_ptr[0]*du_ptr[1];
            sigma_v = dpsis_horiz_ptr[0]*dv_ptr[1];
            sigma_u += dpsis_vert_ptr[0]*du_ptr[stride] + dpsis_vert_ptr[stride_] * du_ptr[stride_];
            sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride] + dpsis_vert_ptr[stride_] * dv_ptr[stride_];
            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

            sigma_u = dpsis_horiz_ptr[stride]*du_ptr[stride + 1];
            sigma_v = dpsis_horiz_ptr[stride]*dv_ptr[stride + 1];
            sigma_u += dpsis_vert_ptr[stride]*du_ptr[stride + stride] + dpsis_vert_ptr[0] * du_ptr[0];
            sigma_v += dpsis_vert_ptr[stride]*dv_ptr[stride + stride] + dpsis_vert_ptr[0] * dv_ptr[0];
            B1 = b1_ptr[stride]+sigma_u;
            B2 = b2_ptr[stride]+sigma_v;
            du_ptr[stride] += omega*( A11_ptr[stride]*B1 + A12_ptr[stride]*B2 - du_ptr[stride] );
            dv_ptr[stride] += omega*( A12_ptr[stride]*B1 + A22_ptr[stride]*B2 - dv_ptr[stride] );

            du_ptr++; dv_ptr++;
            A11_ptr++; A12_ptr++; A22_ptr++;
            b1_ptr++; b2_ptr++;
            dpsis_horiz_ptr++; dpsis_vert_ptr++;
            //count+=2;

            /*    4 5
             * 6 |0 1| 10
             * 7 |2 3| 11  -> first compute 0, then compute 1 & 2 independently, then computer 3
             *    8 9
             * Somehow we need to resolve dependence on 1 & 4 manually.
             * */

            for( i = 0; i < i_block_iter; ++i)
            {
                // Load everything, so memory aliasing won't be a problem
                float dp_vert_0 = dpsis_vert_ptr[0];
                float dp_vert_1 = dpsis_vert_ptr[1];
                float dp_vert_2 = dpsis_vert_ptr[stride];
                float dp_vert_3 = dpsis_vert_ptr[stride + 1];
                float dp_vert_4 = dpsis_vert_ptr[stride_];
                float dp_vert_5 = dpsis_vert_ptr[stride_ + 1];

                float dp_horiz_0 = dpsis_horiz_ptr[0];
                float dp_horiz_1 = dpsis_horiz_ptr[1];
                float dp_horiz_2 = dpsis_horiz_ptr[stride];
                float dp_horiz_3 = dpsis_horiz_ptr[stride + 1];
                float dp_horiz_6 = dpsis_horiz_ptr[-1];
                float dp_horiz_7 = dpsis_horiz_ptr[stride - 1];

                float du_0 = du_ptr[0];
                float du_1 = du_ptr[1];
                float du_2 = du_ptr[stride];
                float du_3 = du_ptr[stride + 1];
                float du_4 = du_ptr[stride_];
                float du_5 = du_ptr[stride_ + 1];
                float du_6 = du_ptr[-1];
                float du_7 = du_ptr[stride - 1];
                float du_8 = du_ptr[stride + stride];
                float du_9 = du_ptr[stride + stride + 1];
                float du_10 = du_ptr[2];
                float du_11 = du_ptr[stride + 2];

                float dv_0 = dv_ptr[0];
                float dv_1 = dv_ptr[1];
                float dv_2 = dv_ptr[stride];
                float dv_3 = dv_ptr[stride + 1];
                float dv_4 = dv_ptr[stride_];
                float dv_5 = dv_ptr[stride_ + 1];
                float dv_6 = dv_ptr[-1];
                float dv_7 = dv_ptr[stride - 1];
                float dv_8 = dv_ptr[stride + stride];
                float dv_9 = dv_ptr[stride + stride + 1];
                float dv_10 = dv_ptr[2];
                float dv_11 = dv_ptr[stride + 2];


                // Load A11, A12, A22, B1, B2
                float b1_0 = b1_ptr[0];
                float b1_1 = b1_ptr[1];
                float b1_2 = b1_ptr[stride];
                float b1_3 = b1_ptr[stride + 1];

                float b2_0 = b2_ptr[0];
                float b2_1 = b2_ptr[1];
                float b2_2 = b2_ptr[stride];
                float b2_3 = b2_ptr[stride + 1];

                float A11_0 = A11_ptr[0];
                float A11_1 = A11_ptr[1];
                float A11_2 = A11_ptr[stride];
                float A11_3 = A11_ptr[stride + 1];

                float A12_0 = A12_ptr[0];
                float A12_1 = A12_ptr[1];
                float A12_2 = A12_ptr[stride];
                float A12_3 = A12_ptr[stride + 1];

                float A22_0 = A22_ptr[0];
                float A22_1 = A22_ptr[1];
                float A22_2 = A22_ptr[stride];
                float A22_3 = A22_ptr[stride + 1];

                // 0
                sigma_u = dp_horiz_6 * du_6 + dp_horiz_0 * du_1 + dp_vert_4 * du_4 + dp_vert_0 * du_2;
                sigma_v = dp_horiz_6 * dv_6 + dp_horiz_0 * dv_1 + dp_vert_4 * dv_4 + dp_vert_0 * dv_2;

                B1 = b1_0 + sigma_u;
                B2 = b2_0 + sigma_v;

                du_0 += omega*( A11_0 * B1 + A12_0 * B2 - du_0 );
                dv_0 += omega*( A12_0 * B1 + A22_0 * B2 - dv_0 );

                // 1 and 2 are independent of each other
                {
                    // 1
                    sigma_u = dp_horiz_0 * du_0 + dp_horiz_1 * du_10 + dp_vert_5 * du_5 + dp_vert_1 * du_3;
                    sigma_v = dp_horiz_0 * dv_0 + dp_horiz_1 * dv_10 + dp_vert_5 * dv_5 + dp_vert_1 * dv_3;

                    B1 = b1_1 + sigma_u;
                    B2 = b2_1 + sigma_v;

                    du_1 += omega*( A11_1 * B1 + A12_1 * B2 - du_1 );
                    dv_1 += omega*( A12_1 * B1 + A22_1 * B2 - dv_1 );

                    // 2
                    sigma_u = dp_horiz_7 * du_7 + dp_horiz_2 * du_3 + dp_vert_0 * du_0 + dp_vert_2 * du_8;
                    sigma_v = dp_horiz_7 * dv_7 + dp_horiz_2 * dv_3 + dp_vert_0 * dv_0 + dp_vert_2 * dv_8;

                    B1 = b1_2 + sigma_u;
                    B2 = b2_2 + sigma_v;

                    du_2 += omega*( A11_2 * B1 + A12_2 * B2 - du_2 );
                    dv_2 += omega*( A12_2 * B1 + A22_2 * B2 - dv_2 );

                }

                // 3
                sigma_u = dp_horiz_2 * du_2 + dp_horiz_3 * du_11 + dp_vert_1 * du_1 + dp_vert_3 * du_9;
                sigma_v = dp_horiz_2 * dv_2 + dp_horiz_3 * dv_11 + dp_vert_1 * dv_1 + dp_vert_3 * dv_9;

                B1 = b1_3 + sigma_u;
                B2 = b2_3 + sigma_v;

                du_3 += omega*( A11_3 * B1 + A12_3 * B2 - du_3 );
                dv_3 += omega*( A12_3 * B1 + A22_3 * B2 - dv_3 );

                // Write back to memory
                du_ptr[0] = du_0;
                du_ptr[1] = du_1;
                du_ptr[stride] = du_2;
                du_ptr[stride + 1] = du_3;
                dv_ptr[0] = dv_0;
                dv_ptr[1] = dv_1;
                dv_ptr[stride] = dv_2;
                dv_ptr[stride + 1] = dv_3;

                du_ptr+=2; dv_ptr+=2;
                A11_ptr+=2; A12_ptr+=2; A22_ptr+=2;
                b1_ptr+=2; b2_ptr+=2;
                dpsis_horiz_ptr+=2; dpsis_vert_ptr+=2;

                //count += 4;
            }

            // Do we need to pad one column or two columns
            if(odd_col)
            {
                //BRK();
                sigma_u = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1];
                sigma_v = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1];
                sigma_u += dpsis_vert_ptr[0]*du_ptr[stride] + dpsis_vert_ptr[stride_] * du_ptr[stride_];
                sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride] + dpsis_vert_ptr[stride_] * dv_ptr[stride_];
                B1 = b1_ptr[0]+sigma_u;
                B2 = b2_ptr[0]+sigma_v;
                du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
                dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

                sigma_u = dpsis_horiz_ptr[stride - 1]*du_ptr[stride - 1] + dpsis_horiz_ptr[stride]*du_ptr[stride + 1];
                sigma_v = dpsis_horiz_ptr[stride - 1]*dv_ptr[stride - 1] + dpsis_horiz_ptr[stride]*dv_ptr[stride + 1];
                sigma_u += dpsis_vert_ptr[stride]*du_ptr[stride + stride] + dpsis_vert_ptr[0] * du_ptr[0];
                sigma_v += dpsis_vert_ptr[stride]*dv_ptr[stride + stride] + dpsis_vert_ptr[0] * dv_ptr[0];
                B1 = b1_ptr[stride]+sigma_u;
                B2 = b2_ptr[stride]+sigma_v;
                du_ptr[stride] += omega*( A11_ptr[stride]*B1 + A12_ptr[stride]*B2 - du_ptr[stride] );
                dv_ptr[stride] += omega*( A12_ptr[stride]*B1 + A22_ptr[stride]*B2 - dv_ptr[stride] );

                du_ptr++; dv_ptr++;
                A11_ptr++; A12_ptr++; A22_ptr++;
                b1_ptr++; b2_ptr++;
                dpsis_horiz_ptr++; dpsis_vert_ptr++;
                //count+=2;

            }

            sigma_u = dpsis_horiz_ptr[-1]*du_ptr[-1];
            sigma_v = dpsis_horiz_ptr[-1]*dv_ptr[-1];
            sigma_u += dpsis_vert_ptr[0]*du_ptr[stride] + dpsis_vert_ptr[stride_] * du_ptr[stride_];
            sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride] + dpsis_vert_ptr[stride_] * dv_ptr[stride_];
            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

            sigma_u = dpsis_horiz_ptr[stride - 1]*du_ptr[stride - 1] + dpsis_horiz_ptr[stride]*du_ptr[stride + 1];
            sigma_v = dpsis_horiz_ptr[stride - 1]*dv_ptr[stride - 1] + dpsis_horiz_ptr[stride]*dv_ptr[stride + 1];
            sigma_u += dpsis_vert_ptr[stride]*du_ptr[stride + stride] + dpsis_vert_ptr[0] * du_ptr[0];
            sigma_v += dpsis_vert_ptr[stride]*dv_ptr[stride + stride] + dpsis_vert_ptr[0] * dv_ptr[0];
            B1 = b1_ptr[stride]+sigma_u;
            B2 = b2_ptr[stride]+sigma_v;
            du_ptr[stride] += omega*( A11_ptr[stride]*B1 + A12_ptr[stride]*B2 - du_ptr[stride] );
            dv_ptr[stride] += omega*( A12_ptr[stride]*B1 + A22_ptr[stride]*B2 - dv_ptr[stride] );

            du_ptr+=block_line_incr; dv_ptr+=block_line_incr;
            A11_ptr+=block_line_incr; A12_ptr+=block_line_incr; A22_ptr+=block_line_incr;
            b1_ptr+=block_line_incr; b2_ptr+=block_line_incr;
            dpsis_horiz_ptr+=block_line_incr; dpsis_vert_ptr+=block_line_incr;
            //count+=2;

        }

        // Check if we need to pad one row or two rows

        int mini_loop = odd_row? 2:1;

        for(int k = mini_loop; k > 0; --k) {
            // left
            sigma_u = dpsis_horiz_ptr[0] * du_ptr[1];
            sigma_v = dpsis_horiz_ptr[0] * dv_ptr[1];
            sigma_u += dpsis_vert_ptr[stride_] * du_ptr[stride_];
            sigma_v += dpsis_vert_ptr[stride_] * dv_ptr[stride_];

            if (k == 2) {
                sigma_u += dpsis_vert_ptr[0] * du_ptr[stride];
                sigma_v += dpsis_vert_ptr[0] * dv_ptr[stride];
            }

            B1 = b1_ptr[0] + sigma_u;
            B2 = b2_ptr[0] + sigma_v;
            du_ptr[0] += omega * (A11_ptr[0] * B1 + A12_ptr[0] * B2 - du_ptr[0]);
            dv_ptr[0] += omega * (A12_ptr[0] * B1 + A22_ptr[0] * B2 - dv_ptr[0]);

            du_ptr++;
            dv_ptr++;
            A11_ptr++;
            A12_ptr++;
            A22_ptr++;
            b1_ptr++;
            b2_ptr++;
            dpsis_horiz_ptr++;
            dpsis_vert_ptr++;
            //count++;

            // middle of the first line
            for (i = 1; i < du->width - 1; ++i) {
                sigma_u = dpsis_horiz_ptr[-1] * du_ptr[-1] + dpsis_horiz_ptr[0] * du_ptr[1];
                sigma_v = dpsis_horiz_ptr[-1] * dv_ptr[-1] + dpsis_horiz_ptr[0] * dv_ptr[1];
                sigma_u += dpsis_vert_ptr[stride_] * du_ptr[stride_];
                sigma_v += dpsis_vert_ptr[stride_] * dv_ptr[stride_];

                if (k == 2) {
                    sigma_u += dpsis_vert_ptr[0] * du_ptr[stride];
                    sigma_v += dpsis_vert_ptr[0] * dv_ptr[stride];
                }

                B1 = b1_ptr[0] + sigma_u;
                B2 = b2_ptr[0] + sigma_v;
                du_ptr[0] += omega * (A11_ptr[0] * B1 + A12_ptr[0] * B2 - du_ptr[0]);
                dv_ptr[0] += omega * (A12_ptr[0] * B1 + A22_ptr[0] * B2 - dv_ptr[0]);

                du_ptr++;
                dv_ptr++;
                A11_ptr++;
                A12_ptr++;
                A22_ptr++;
                b1_ptr++;
                b2_ptr++;
                dpsis_horiz_ptr++;
                dpsis_vert_ptr++;
                //count++;
            }

            // right
            sigma_u = dpsis_horiz_ptr[-1] * du_ptr[-1];
            sigma_v = dpsis_horiz_ptr[-1] * dv_ptr[-1];
            sigma_u += dpsis_vert_ptr[stride_] * du_ptr[stride_];
            sigma_v += dpsis_vert_ptr[stride_] * dv_ptr[stride_];

            if (k == 2) {
                sigma_u += dpsis_vert_ptr[0] * du_ptr[stride];
                sigma_v += dpsis_vert_ptr[0] * dv_ptr[stride];
            }

            B1 = b1_ptr[0] + sigma_u;
            B2 = b2_ptr[0] + sigma_v;
            du_ptr[0] += omega * (A11_ptr[0] * B1 + A12_ptr[0] * B2 - du_ptr[0]);
            dv_ptr[0] += omega * (A12_ptr[0] * B1 + A22_ptr[0] * B2 - dv_ptr[0]);

            //send the pointer to next line

            du_ptr += new_line_incr;
            dv_ptr += new_line_incr;
            A11_ptr += new_line_incr;
            A12_ptr += new_line_incr;
            A22_ptr += new_line_incr;
            b1_ptr += new_line_incr;
            b2_ptr += new_line_incr;
            dpsis_horiz_ptr += new_line_incr;
            dpsis_vert_ptr += new_line_incr;
            //count++;

        }

    }


    image_delete(A11); image_delete(A12); image_delete(A22);

}

void sor_coupled_blocked_2x2_vectorization(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, int iterations, float omega)
{
    // Fall back to standard solver in case of trivial cases
    if(du->width < 2 || du->height < 2 || iterations < 1)
    {
        sor_coupled_slow_but_readable(du,dv,a11,a12,a22,b1,b2,dpsis_horiz,dpsis_vert,iterations, omega);
        return;
    }

    int i, j, iter;
    int stride = du->stride;
    int stride_ = -stride;
    //int stride_1 = stride_ + 1;
    //int stride_2 = stride_ + 2;
    //int stride_3 = stride_ + 3;


    float *du_ptr = du->data, *dv_ptr = dv->data;
    float *a11_ptr = a11->data, *a12_ptr = a12->data, *a22_ptr = a22->data;
    float *b1_ptr = b1->data, *b2_ptr = b2->data;
    float *dpsis_horiz_ptr = dpsis_horiz->data, *dpsis_vert_ptr = dpsis_vert->data;

    // Typical exchage memory for speed
    image_t *A11 = image_new(du->width,du->height);
    image_t *A22 = image_new(du->width,du->height);
    image_t *A12 = image_new(du->width,du->height);

    float *A11_ptr = A11->data, *A12_ptr = A12->data, *A22_ptr = A22->data;

    float sum_dpsis, sigma_u, sigma_v, B1, B2, det;
    float sum_dpsis_1; // sum_dpsis_2, sum_dpsis_3, sum_dpsis_4;
    // float sigma_u_1, sigma_u_2, sigma_u_3, sigma_u_4;
    // float sigma_v_1, sigma_v_2, sigma_v_3, sigma_v_4;

    // Should be able to hold 3 blocks simultaneously.
    int bsize = 4;
    // int npad_w = (du->width - 2) % bsize;
    int niter_w = (du->width - 2) / bsize;
    // int npad_h = (du->height - 2) % bsize;
    // int niter_h = (du->height - 2) / bsize;

    int i_bound = bsize * niter_w + 1;
    // int j_bound = bsize * niter_h + 1;

    int new_line_incr = du->stride - du->width + 1;

    float A11_;
    float A12_;
    float A22_;

    // variables for SSE operations
    __m128 dpsis_horiz_pl, dpsis_horiz_pr, dpsis_vert_pu, dpsis_vert_pl;
    __m128 sum_dpsis_p;
    __m128 A11_p, A12_p, A22_p;
    __m128 zeros = _mm_setzero_ps();
    __m128 det_1_p, det_2_p, det_p;

    for (j = 0; j < du->height; ++j) {

        // Left cell
        sum_dpsis = dpsis_horiz_ptr[0];

        if(j > 0)
            sum_dpsis += dpsis_vert_ptr[stride_];
        if (j < du->height - 1)
            sum_dpsis += dpsis_vert_ptr[0];

        A22_ = a11_ptr[0]+sum_dpsis;
        A11_ = a22_ptr[0]+sum_dpsis;
        A12_ = -a12_ptr[0];

        det = A11_*A22_ - A12_*A12_;

        A11_ptr[0] = A11_ / det;
        A22_ptr[0] = A22_ / det;
        A12_ptr[0] = A12_ / det;

        dpsis_horiz_ptr++; dpsis_vert_ptr++;
        A11_ptr++; A12_ptr++; A22_ptr++;
        a11_ptr++; a12_ptr++; a22_ptr++;

        // first row
        for (i = 1;  i < i_bound; i+=4) {
            // Column 1

            dpsis_horiz_pr = _mm_loadu_ps(dpsis_horiz_ptr);
            dpsis_horiz_pl = _mm_loadu_ps(dpsis_horiz_ptr - 1);

            sum_dpsis_p = _mm_add_ps(dpsis_horiz_pl, dpsis_horiz_pr);

            if(j > 0)
            {
                dpsis_vert_pl = _mm_loadu_ps(dpsis_vert_ptr + stride_);
                sum_dpsis_p = _mm_add_ps(sum_dpsis_p, dpsis_vert_pl);
            }

            if(j < du->height - 1)
            {
                dpsis_vert_pu = _mm_loadu_ps(dpsis_vert_ptr);
                sum_dpsis_p = _mm_add_ps(sum_dpsis_p, dpsis_vert_pu);
            }


            sum_dpsis_1 = dpsis_horiz_ptr[-1] + dpsis_horiz_ptr[0];

            if(j > 0)
                sum_dpsis_1 += dpsis_vert_ptr[stride_];
            if (j < du->height - 1)
                sum_dpsis_1 += dpsis_vert_ptr[0];

            A22_p = _mm_loadu_ps(a11_ptr);
            A11_p = _mm_loadu_ps(a22_ptr);
            A12_p = _mm_loadu_ps(a12_ptr);

            A22_p = _mm_add_ps(A22_p, sum_dpsis_p);
            A11_p = _mm_add_ps(A11_p, sum_dpsis_p);
            A12_p = _mm_sub_ps(zeros, A12_p);

            det_1_p = _mm_mul_ps(A11_p, A22_p);
            det_2_p = _mm_mul_ps(A12_p, A12_p);
            det_p = _mm_sub_ps(det_1_p, det_2_p);

            A22_p = _mm_div_ps(A22_p, det_p);
            A11_p = _mm_div_ps(A11_p, det_p);
            A12_p = _mm_div_ps(A12_p, det_p);

            //printf("%d", (int)(A11_ptr - A11->data));
            _mm_storeu_ps(A11_ptr, A11_p);
            _mm_storeu_ps(A12_ptr, A12_p);
            _mm_storeu_ps(A22_ptr, A22_p);

            dpsis_horiz_ptr+=4; dpsis_vert_ptr+=4;
            A11_ptr+=4; A12_ptr+=4; A22_ptr+=4;
            a11_ptr+=4; a12_ptr+=4; a22_ptr+=4;
        }

        // Cope with whatever that's left
        while(i < du->width - 1)
        {
            sum_dpsis = dpsis_horiz_ptr[-1] + dpsis_horiz_ptr[0];

            if(j > 0)
                sum_dpsis += dpsis_vert_ptr[stride_];
            if (j < du->height - 1)
                sum_dpsis += dpsis_vert_ptr[0];

            A22_ = a11_ptr[0]+sum_dpsis;
            A11_ = a22_ptr[0]+sum_dpsis;
            A12_ = -a12_ptr[0];

            det = A11_*A22_ - A12_*A12_;

            A11_ptr[0] = A11_ / det;
            A22_ptr[0] = A22_ / det;
            A12_ptr[0] = A12_ / det;

            dpsis_horiz_ptr++; dpsis_vert_ptr++;
            A11_ptr++; A12_ptr++; A22_ptr++;
            a11_ptr++; a12_ptr++; a22_ptr++;
            i++;
        }

        // upperright corner
        sum_dpsis = dpsis_horiz_ptr[-1];

        if(j > 0)
            sum_dpsis += dpsis_vert_ptr[stride_];
        if (j < du->height - 1)
            sum_dpsis += dpsis_vert_ptr[0];

        A22_ = a11_ptr[0]+sum_dpsis;
        A11_ = a22_ptr[0]+sum_dpsis;
        A12_ = -a12_ptr[0];

        det = A11_*A22_ - A12_*A12_;

        A11_ptr[0] = A11_ / det;
        A22_ptr[0] = A22_ / det;
        A12_ptr[0] = A12_ / det;

        dpsis_horiz_ptr += new_line_incr;
        dpsis_vert_ptr += new_line_incr;
        A11_ptr+=new_line_incr; A12_ptr+=new_line_incr; A22_ptr+=new_line_incr;
        a11_ptr+=new_line_incr; a12_ptr+=new_line_incr; a22_ptr+=new_line_incr;
    }

    // Main iterations
    // Loop over 2x2 blocks. Special cases: left/right column, top/bottom row, possible one more extra row/column.
    int block_line_incr = du->stride + (du->stride - du->width + 1);
    int j_block_iter = (du->height - 2) / 2;
    int i_block_iter = (du->width - 2) / 2;
    bool odd_row = du->height % 2? true:false;
    bool odd_col = du->width % 2? true:false;

    for(iter = 0; iter < iterations; ++iter)
    {

        //int count = 0;
        // set pointer to the beginning
        du_ptr = du->data; dv_ptr = dv->data;
        A11_ptr = A11->data; A12_ptr = A12->data; A22_ptr = A22->data;
        b1_ptr = b1->data; b2_ptr = b2->data;
        dpsis_horiz_ptr = dpsis_horiz->data; dpsis_vert_ptr = dpsis_vert->data;

        // upperleft corner
        sigma_u = dpsis_horiz_ptr[0]*du_ptr[1];
        sigma_v = dpsis_horiz_ptr[0]*dv_ptr[1];
        sigma_u += dpsis_vert_ptr[0]*du_ptr[stride];
        sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride];
        B1 = b1_ptr[0]+sigma_u;
        B2 = b2_ptr[0]+sigma_v;
        du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
        dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

        du_ptr++; dv_ptr++;
        A11_ptr++; A12_ptr++; A22_ptr++;
        b1_ptr++; b2_ptr++;
        dpsis_horiz_ptr++; dpsis_vert_ptr++;
        // count++;
        // middle of the first line
        for( i = 1; i < du->width - 1; i++)
        {
            sigma_u = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1];
            sigma_v = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1];
            sigma_u += dpsis_vert_ptr[0]*du_ptr[stride];
            sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride];

            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

            du_ptr++; dv_ptr++;
            A11_ptr++; A12_ptr++; A22_ptr++;
            b1_ptr++; b2_ptr++;
            dpsis_horiz_ptr++; dpsis_vert_ptr++;
            //count++;
        }

        // rightupper corner
        sigma_u = dpsis_horiz_ptr[-1]*du_ptr[-1];
        sigma_v = dpsis_horiz_ptr[-1]*dv_ptr[-1];
        sigma_u += dpsis_vert_ptr[0]*du_ptr[stride];
        sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride];
        B1 = b1_ptr[0]+sigma_u;
        B2 = b2_ptr[0]+sigma_v;
        du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
        dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

        //send the pointer to next line
        du_ptr+=new_line_incr; dv_ptr+=new_line_incr;
        b1_ptr+=new_line_incr; b2_ptr+=new_line_incr;
        dpsis_horiz_ptr += new_line_incr; dpsis_vert_ptr += new_line_incr;
        A11_ptr+=new_line_incr; A12_ptr+=new_line_incr; A22_ptr+=new_line_incr;

        //count++;
        for(j = 0; j < j_block_iter; ++j)
        {

            // First column
            sigma_u = dpsis_horiz_ptr[0]*du_ptr[1];
            sigma_v = dpsis_horiz_ptr[0]*dv_ptr[1];
            sigma_u += dpsis_vert_ptr[0]*du_ptr[stride] + dpsis_vert_ptr[stride_] * du_ptr[stride_];
            sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride] + dpsis_vert_ptr[stride_] * dv_ptr[stride_];
            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

            sigma_u = dpsis_horiz_ptr[stride]*du_ptr[stride + 1];
            sigma_v = dpsis_horiz_ptr[stride]*dv_ptr[stride + 1];
            sigma_u += dpsis_vert_ptr[stride]*du_ptr[stride + stride] + dpsis_vert_ptr[0] * du_ptr[0];
            sigma_v += dpsis_vert_ptr[stride]*dv_ptr[stride + stride] + dpsis_vert_ptr[0] * dv_ptr[0];
            B1 = b1_ptr[stride]+sigma_u;
            B2 = b2_ptr[stride]+sigma_v;
            du_ptr[stride] += omega*( A11_ptr[stride]*B1 + A12_ptr[stride]*B2 - du_ptr[stride] );
            dv_ptr[stride] += omega*( A12_ptr[stride]*B1 + A22_ptr[stride]*B2 - dv_ptr[stride] );

            du_ptr++; dv_ptr++;
            A11_ptr++; A12_ptr++; A22_ptr++;
            b1_ptr++; b2_ptr++;
            dpsis_horiz_ptr++; dpsis_vert_ptr++;
            //count+=2;

            /*    4 5
             * 6 |0 1| 10
             * 7 |2 3| 11  -> first compute 0, then compute 1 & 2 independently, then computer 3
             *    8 9
             * Somehow we need to resolve dependence on 1 & 4 manually.
             * */

            for( i = 0; i < i_block_iter; ++i)
            {
                // Load everything, so memory aliasing won't be a problem
                float dp_vert_0 = dpsis_vert_ptr[0];
                float dp_vert_1 = dpsis_vert_ptr[1];
                float dp_vert_2 = dpsis_vert_ptr[stride];
                float dp_vert_3 = dpsis_vert_ptr[stride + 1];
                float dp_vert_4 = dpsis_vert_ptr[stride_];
                float dp_vert_5 = dpsis_vert_ptr[stride_ + 1];

                float dp_horiz_0 = dpsis_horiz_ptr[0];
                float dp_horiz_1 = dpsis_horiz_ptr[1];
                float dp_horiz_2 = dpsis_horiz_ptr[stride];
                float dp_horiz_3 = dpsis_horiz_ptr[stride + 1];
                float dp_horiz_6 = dpsis_horiz_ptr[-1];
                float dp_horiz_7 = dpsis_horiz_ptr[stride - 1];

                float du_0 = du_ptr[0];
                float du_1 = du_ptr[1];
                float du_2 = du_ptr[stride];
                float du_3 = du_ptr[stride + 1];
                float du_4 = du_ptr[stride_];
                float du_5 = du_ptr[stride_ + 1];
                float du_6 = du_ptr[-1];
                float du_7 = du_ptr[stride - 1];
                float du_8 = du_ptr[stride + stride];
                float du_9 = du_ptr[stride + stride + 1];
                float du_10 = du_ptr[2];
                float du_11 = du_ptr[stride + 2];

                float dv_0 = dv_ptr[0];
                float dv_1 = dv_ptr[1];
                float dv_2 = dv_ptr[stride];
                float dv_3 = dv_ptr[stride + 1];
                float dv_4 = dv_ptr[stride_];
                float dv_5 = dv_ptr[stride_ + 1];
                float dv_6 = dv_ptr[-1];
                float dv_7 = dv_ptr[stride - 1];
                float dv_8 = dv_ptr[stride + stride];
                float dv_9 = dv_ptr[stride + stride + 1];
                float dv_10 = dv_ptr[2];
                float dv_11 = dv_ptr[stride + 2];


                // Load A11, A12, A22, B1, B2
                float b1_0 = b1_ptr[0];
                float b1_1 = b1_ptr[1];
                float b1_2 = b1_ptr[stride];
                float b1_3 = b1_ptr[stride + 1];

                float b2_0 = b2_ptr[0];
                float b2_1 = b2_ptr[1];
                float b2_2 = b2_ptr[stride];
                float b2_3 = b2_ptr[stride + 1];

                float A11_0 = A11_ptr[0];
                float A11_1 = A11_ptr[1];
                float A11_2 = A11_ptr[stride];
                float A11_3 = A11_ptr[stride + 1];

                float A12_0 = A12_ptr[0];
                float A12_1 = A12_ptr[1];
                float A12_2 = A12_ptr[stride];
                float A12_3 = A12_ptr[stride + 1];

                float A22_0 = A22_ptr[0];
                float A22_1 = A22_ptr[1];
                float A22_2 = A22_ptr[stride];
                float A22_3 = A22_ptr[stride + 1];

                // 0
                sigma_u = dp_horiz_6 * du_6 + dp_horiz_0 * du_1 + dp_vert_4 * du_4 + dp_vert_0 * du_2;
                sigma_v = dp_horiz_6 * dv_6 + dp_horiz_0 * dv_1 + dp_vert_4 * dv_4 + dp_vert_0 * dv_2;

                B1 = b1_0 + sigma_u;
                B2 = b2_0 + sigma_v;

                du_0 += omega*( A11_0 * B1 + A12_0 * B2 - du_0 );
                dv_0 += omega*( A12_0 * B1 + A22_0 * B2 - dv_0 );

                // 1 and 2 are independent of each other
                {
                    // 1
                    sigma_u = dp_horiz_0 * du_0 + dp_horiz_1 * du_10 + dp_vert_5 * du_5 + dp_vert_1 * du_3;
                    sigma_v = dp_horiz_0 * dv_0 + dp_horiz_1 * dv_10 + dp_vert_5 * dv_5 + dp_vert_1 * dv_3;

                    B1 = b1_1 + sigma_u;
                    B2 = b2_1 + sigma_v;

                    du_1 += omega*( A11_1 * B1 + A12_1 * B2 - du_1 );
                    dv_1 += omega*( A12_1 * B1 + A22_1 * B2 - dv_1 );

                    // 2
                    sigma_u = dp_horiz_7 * du_7 + dp_horiz_2 * du_3 + dp_vert_0 * du_0 + dp_vert_2 * du_8;
                    sigma_v = dp_horiz_7 * dv_7 + dp_horiz_2 * dv_3 + dp_vert_0 * dv_0 + dp_vert_2 * dv_8;

                    B1 = b1_2 + sigma_u;
                    B2 = b2_2 + sigma_v;

                    du_2 += omega*( A11_2 * B1 + A12_2 * B2 - du_2 );
                    dv_2 += omega*( A12_2 * B1 + A22_2 * B2 - dv_2 );

                }

                // 3
                sigma_u = dp_horiz_2 * du_2 + dp_horiz_3 * du_11 + dp_vert_1 * du_1 + dp_vert_3 * du_9;
                sigma_v = dp_horiz_2 * dv_2 + dp_horiz_3 * dv_11 + dp_vert_1 * dv_1 + dp_vert_3 * dv_9;

                B1 = b1_3 + sigma_u;
                B2 = b2_3 + sigma_v;

                du_3 += omega*( A11_3 * B1 + A12_3 * B2 - du_3 );
                dv_3 += omega*( A12_3 * B1 + A22_3 * B2 - dv_3 );

                // Write back to memory
                du_ptr[0] = du_0;
                du_ptr[1] = du_1;
                du_ptr[stride] = du_2;
                du_ptr[stride + 1] = du_3;
                dv_ptr[0] = dv_0;
                dv_ptr[1] = dv_1;
                dv_ptr[stride] = dv_2;
                dv_ptr[stride + 1] = dv_3;

                du_ptr+=2; dv_ptr+=2;
                A11_ptr+=2; A12_ptr+=2; A22_ptr+=2;
                b1_ptr+=2; b2_ptr+=2;
                dpsis_horiz_ptr+=2; dpsis_vert_ptr+=2;

                //count += 4;
            }

            // Do we need to pad one column or two columns
            if(odd_col)
            {
                //BRK();
                sigma_u = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1];
                sigma_v = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1];
                sigma_u += dpsis_vert_ptr[0]*du_ptr[stride] + dpsis_vert_ptr[stride_] * du_ptr[stride_];
                sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride] + dpsis_vert_ptr[stride_] * dv_ptr[stride_];
                B1 = b1_ptr[0]+sigma_u;
                B2 = b2_ptr[0]+sigma_v;
                du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
                dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

                sigma_u = dpsis_horiz_ptr[stride - 1]*du_ptr[stride - 1] + dpsis_horiz_ptr[stride]*du_ptr[stride + 1];
                sigma_v = dpsis_horiz_ptr[stride - 1]*dv_ptr[stride - 1] + dpsis_horiz_ptr[stride]*dv_ptr[stride + 1];
                sigma_u += dpsis_vert_ptr[stride]*du_ptr[stride + stride] + dpsis_vert_ptr[0] * du_ptr[0];
                sigma_v += dpsis_vert_ptr[stride]*dv_ptr[stride + stride] + dpsis_vert_ptr[0] * dv_ptr[0];
                B1 = b1_ptr[stride]+sigma_u;
                B2 = b2_ptr[stride]+sigma_v;
                du_ptr[stride] += omega*( A11_ptr[stride]*B1 + A12_ptr[stride]*B2 - du_ptr[stride] );
                dv_ptr[stride] += omega*( A12_ptr[stride]*B1 + A22_ptr[stride]*B2 - dv_ptr[stride] );

                du_ptr++; dv_ptr++;
                A11_ptr++; A12_ptr++; A22_ptr++;
                b1_ptr++; b2_ptr++;
                dpsis_horiz_ptr++; dpsis_vert_ptr++;
                //count+=2;

            }

            sigma_u = dpsis_horiz_ptr[-1]*du_ptr[-1];
            sigma_v = dpsis_horiz_ptr[-1]*dv_ptr[-1];
            sigma_u += dpsis_vert_ptr[0]*du_ptr[stride] + dpsis_vert_ptr[stride_] * du_ptr[stride_];
            sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride] + dpsis_vert_ptr[stride_] * dv_ptr[stride_];
            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

            sigma_u = dpsis_horiz_ptr[stride - 1]*du_ptr[stride - 1] + dpsis_horiz_ptr[stride]*du_ptr[stride + 1];
            sigma_v = dpsis_horiz_ptr[stride - 1]*dv_ptr[stride - 1] + dpsis_horiz_ptr[stride]*dv_ptr[stride + 1];
            sigma_u += dpsis_vert_ptr[stride]*du_ptr[stride + stride] + dpsis_vert_ptr[0] * du_ptr[0];
            sigma_v += dpsis_vert_ptr[stride]*dv_ptr[stride + stride] + dpsis_vert_ptr[0] * dv_ptr[0];
            B1 = b1_ptr[stride]+sigma_u;
            B2 = b2_ptr[stride]+sigma_v;
            du_ptr[stride] += omega*( A11_ptr[stride]*B1 + A12_ptr[stride]*B2 - du_ptr[stride] );
            dv_ptr[stride] += omega*( A12_ptr[stride]*B1 + A22_ptr[stride]*B2 - dv_ptr[stride] );

            du_ptr+=block_line_incr; dv_ptr+=block_line_incr;
            A11_ptr+=block_line_incr; A12_ptr+=block_line_incr; A22_ptr+=block_line_incr;
            b1_ptr+=block_line_incr; b2_ptr+=block_line_incr;
            dpsis_horiz_ptr+=block_line_incr; dpsis_vert_ptr+=block_line_incr;
            //count+=2;

        }

        // Check if we need to pad one row or two rows

        int mini_loop = odd_row? 2:1;

        for(int k = mini_loop; k > 0; --k) {
            // left
            sigma_u = dpsis_horiz_ptr[0] * du_ptr[1];
            sigma_v = dpsis_horiz_ptr[0] * dv_ptr[1];
            sigma_u += dpsis_vert_ptr[stride_] * du_ptr[stride_];
            sigma_v += dpsis_vert_ptr[stride_] * dv_ptr[stride_];

            if (k == 2) {
                sigma_u += dpsis_vert_ptr[0] * du_ptr[stride];
                sigma_v += dpsis_vert_ptr[0] * dv_ptr[stride];
            }

            B1 = b1_ptr[0] + sigma_u;
            B2 = b2_ptr[0] + sigma_v;
            du_ptr[0] += omega * (A11_ptr[0] * B1 + A12_ptr[0] * B2 - du_ptr[0]);
            dv_ptr[0] += omega * (A12_ptr[0] * B1 + A22_ptr[0] * B2 - dv_ptr[0]);

            du_ptr++;
            dv_ptr++;
            A11_ptr++;
            A12_ptr++;
            A22_ptr++;
            b1_ptr++;
            b2_ptr++;
            dpsis_horiz_ptr++;
            dpsis_vert_ptr++;
            //count++;

            // middle of the first line
            for (i = 1; i < du->width - 1; ++i) {
                sigma_u = dpsis_horiz_ptr[-1] * du_ptr[-1] + dpsis_horiz_ptr[0] * du_ptr[1];
                sigma_v = dpsis_horiz_ptr[-1] * dv_ptr[-1] + dpsis_horiz_ptr[0] * dv_ptr[1];
                sigma_u += dpsis_vert_ptr[stride_] * du_ptr[stride_];
                sigma_v += dpsis_vert_ptr[stride_] * dv_ptr[stride_];

                if (k == 2) {
                    sigma_u += dpsis_vert_ptr[0] * du_ptr[stride];
                    sigma_v += dpsis_vert_ptr[0] * dv_ptr[stride];
                }

                B1 = b1_ptr[0] + sigma_u;
                B2 = b2_ptr[0] + sigma_v;
                du_ptr[0] += omega * (A11_ptr[0] * B1 + A12_ptr[0] * B2 - du_ptr[0]);
                dv_ptr[0] += omega * (A12_ptr[0] * B1 + A22_ptr[0] * B2 - dv_ptr[0]);

                du_ptr++;
                dv_ptr++;
                A11_ptr++;
                A12_ptr++;
                A22_ptr++;
                b1_ptr++;
                b2_ptr++;
                dpsis_horiz_ptr++;
                dpsis_vert_ptr++;
                //count++;
            }

            // right
            sigma_u = dpsis_horiz_ptr[-1] * du_ptr[-1];
            sigma_v = dpsis_horiz_ptr[-1] * dv_ptr[-1];
            sigma_u += dpsis_vert_ptr[stride_] * du_ptr[stride_];
            sigma_v += dpsis_vert_ptr[stride_] * dv_ptr[stride_];

            if (k == 2) {
                sigma_u += dpsis_vert_ptr[0] * du_ptr[stride];
                sigma_v += dpsis_vert_ptr[0] * dv_ptr[stride];
            }

            B1 = b1_ptr[0] + sigma_u;
            B2 = b2_ptr[0] + sigma_v;
            du_ptr[0] += omega * (A11_ptr[0] * B1 + A12_ptr[0] * B2 - du_ptr[0]);
            dv_ptr[0] += omega * (A12_ptr[0] * B1 + A22_ptr[0] * B2 - dv_ptr[0]);

            //send the pointer to next line

            du_ptr += new_line_incr;
            dv_ptr += new_line_incr;
            A11_ptr += new_line_incr;
            A12_ptr += new_line_incr;
            A22_ptr += new_line_incr;
            b1_ptr += new_line_incr;
            b2_ptr += new_line_incr;
            dpsis_horiz_ptr += new_line_incr;
            dpsis_vert_ptr += new_line_incr;
            //count++;

        }

    }


    image_delete(A11); image_delete(A12); image_delete(A22);

}


void sor_coupled_blocked_4x4(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, int iterations, float omega)
{
    // Fall back to standard solver in case of trivial cases
    if(du->width < 2 || du->height < 2 || iterations < 1)
    {
        sor_coupled_slow_but_readable(du,dv,a11,a12,a22,b1,b2,dpsis_horiz,dpsis_vert,iterations, omega);
        return;
    }

    int i, j, iter;
    int stride = du->stride;
    int stride_ = -stride;
    int stride_1 = stride_ + 1;
    int stride_2 = stride_ + 2;
    int stride_3 = stride_ + 3;


    float *du_ptr = du->data, *dv_ptr = dv->data;
    float *a11_ptr = a11->data, *a12_ptr = a12->data, *a22_ptr = a22->data;
    float *b1_ptr = b1->data, *b2_ptr = b2->data;
    float *dpsis_horiz_ptr = dpsis_horiz->data, *dpsis_vert_ptr = dpsis_vert->data;

    // Typical exchage memory for speed
    image_t *A11 = image_new(du->width,du->height);
    image_t *A22 = image_new(du->width,du->height);
    image_t *A12 = image_new(du->width,du->height);

    float *A11_ptr = A11->data, *A12_ptr = A12->data, *A22_ptr = A22->data;

    float sum_dpsis, sigma_u, sigma_v, B1, B2, det;
    float sum_dpsis_1, sum_dpsis_2, sum_dpsis_3, sum_dpsis_4;

    // Should be able to hold 3 blocks simultaneously.
    int bsize = 4;
    int niter_w = (du->width - 2) / bsize;

    int i_bound = bsize * niter_w + 1;

    int new_line_incr = du->stride - du->width + 1;

    float A11_1, A11_2, A11_3, A11_4, A11_;
    float A12_1, A12_2, A12_3, A12_4, A12_;
    float A22_1, A22_2, A22_3, A22_4, A22_;

    for (j = 0; j < du->height; ++j) {

        // Left cell
        sum_dpsis = dpsis_horiz_ptr[0];

        if(j > 0)
            sum_dpsis += dpsis_vert_ptr[stride_];
        if (j < du->height - 1)
            sum_dpsis += dpsis_vert_ptr[0];

        A22_ = a11_ptr[0]+sum_dpsis;
        A11_ = a22_ptr[0]+sum_dpsis;
        A12_ = -a12_ptr[0];

        det = A11_*A22_ - A12_*A12_;

        A11_ptr[0] = A11_ / det;
        A22_ptr[0] = A22_ / det;
        A12_ptr[0] = A12_ / det;

        dpsis_horiz_ptr++; dpsis_vert_ptr++;
        A11_ptr++; A12_ptr++; A22_ptr++;
        a11_ptr++; a12_ptr++; a22_ptr++;

        // first row
        for (i = 1;  i < i_bound; i+=4) {
            // Column 1
            sum_dpsis_1 = dpsis_horiz_ptr[-1] + dpsis_horiz_ptr[0];

            if(j > 0)
                sum_dpsis_1 += dpsis_vert_ptr[stride_];
            if (j < du->height - 1)
                sum_dpsis_1 += dpsis_vert_ptr[0];

            A22_1 = a11_ptr[0]+sum_dpsis_1;
            A11_1 = a22_ptr[0]+sum_dpsis_1;
            A12_1 = -a12_ptr[0];

            det = A11_1*A22_1 - A12_1*A12_1;

            A11_ptr[0] = A11_1 / det;
            A22_ptr[0] = A22_1 / det;
            A12_ptr[0] = A12_1 / det;

            // Column 2
            sum_dpsis_2 = dpsis_horiz_ptr[0] + dpsis_horiz_ptr[1];

            if(j > 0)
                sum_dpsis_2 += dpsis_vert_ptr[stride_1];
            if (j < du->height - 1)
                sum_dpsis_2 += dpsis_vert_ptr[1];

            A22_2 = a11_ptr[1]+sum_dpsis_2;
            A11_2 = a22_ptr[1]+sum_dpsis_2;
            A12_2 = -a12_ptr[1];

            det = A11_2*A22_2 - A12_2*A12_2;

            A11_ptr[1] = A11_2 / det;
            A22_ptr[1] = A22_2 / det;
            A12_ptr[1] = A12_2 / det;

            // Column 3
            sum_dpsis_3 = dpsis_horiz_ptr[1] + dpsis_horiz_ptr[2];

            if(j > 0)
                sum_dpsis_3 += dpsis_vert_ptr[stride_2];
            if (j < du->height - 1)
                sum_dpsis_3 += dpsis_vert_ptr[2];

            A22_3 = a11_ptr[2]+sum_dpsis_3;
            A11_3 = a22_ptr[2]+sum_dpsis_3;
            A12_3 = -a12_ptr[2];

            det = A11_3*A22_3 - A12_3*A12_3;

            A11_ptr[2] = A11_3 / det;
            A22_ptr[2] = A22_3 / det;
            A12_ptr[2] = A12_3 / det;

            // Column 4
            sum_dpsis_4 = dpsis_horiz_ptr[2] + dpsis_horiz_ptr[3];

            if(j > 0)
                sum_dpsis_4 += dpsis_vert_ptr[stride_3];
            if (j < du->height - 1)
                sum_dpsis_4 += dpsis_vert_ptr[3];

            A22_4 = a11_ptr[3]+sum_dpsis_4;
            A11_4 = a22_ptr[3]+sum_dpsis_4;
            A12_4 = -a12_ptr[3];

            det = A11_4*A22_4 - A12_4*A12_4;

            A11_ptr[3] = A11_4 / det;
            A22_ptr[3] = A22_4 / det;
            A12_ptr[3] = A12_4 / det;

            dpsis_horiz_ptr+=4; dpsis_vert_ptr+=4;
            A11_ptr+=4; A12_ptr+=4; A22_ptr+=4;
            a11_ptr+=4; a12_ptr+=4; a22_ptr+=4;
        }

        // Cope with whatever that's left
        while(i < du->width - 1)
        {
            sum_dpsis = dpsis_horiz_ptr[-1] + dpsis_horiz_ptr[0];

            if(j > 0)
                sum_dpsis += dpsis_vert_ptr[stride_];
            if (j < du->height - 1)
                sum_dpsis += dpsis_vert_ptr[0];

            A22_ = a11_ptr[0]+sum_dpsis;
            A11_ = a22_ptr[0]+sum_dpsis;
            A12_ = -a12_ptr[0];

            det = A11_*A22_ - A12_*A12_;

            A11_ptr[0] = A11_ / det;
            A22_ptr[0] = A22_ / det;
            A12_ptr[0] = A12_ / det;

            dpsis_horiz_ptr++; dpsis_vert_ptr++;
            A11_ptr++; A12_ptr++; A22_ptr++;
            a11_ptr++; a12_ptr++; a22_ptr++;
            i++;
        }

        // upperright corner
        sum_dpsis = dpsis_horiz_ptr[-1];

        if(j > 0)
            sum_dpsis += dpsis_vert_ptr[stride_];
        if (j < du->height - 1)
            sum_dpsis += dpsis_vert_ptr[0];

        A22_ = a11_ptr[0]+sum_dpsis;
        A11_ = a22_ptr[0]+sum_dpsis;
        A12_ = -a12_ptr[0];

        det = A11_*A22_ - A12_*A12_;

        A11_ptr[0] = A11_ / det;
        A22_ptr[0] = A22_ / det;
        A12_ptr[0] = A12_ / det;

        dpsis_horiz_ptr += new_line_incr;
        dpsis_vert_ptr += new_line_incr;
        A11_ptr+=new_line_incr; A12_ptr+=new_line_incr; A22_ptr+=new_line_incr;
        a11_ptr+=new_line_incr; a12_ptr+=new_line_incr; a22_ptr+=new_line_incr;
    }

    // Main iterations
    // Loop over 4x4 blocks. Special cases: left/right column, top/bottom row, possible extra 1-3 rows/columns.
    int block_line_incr = du->stride*4 - du->width + 1;
    int j_block_iter = (du->height - 2) / 4;
    int i_block_iter = (du->width - 2) / 4;
    int extra_row = (du->height - 2) % 4;
    int extra_col = (du->width -2 ) % 4;

    for(iter = 0; iter < iterations; ++iter)
    {
        // set pointer to the beginning
        du_ptr = du->data; dv_ptr = dv->data;
        A11_ptr = A11->data; A12_ptr = A12->data; A22_ptr = A22->data;
        b1_ptr = b1->data; b2_ptr = b2->data;
        dpsis_horiz_ptr = dpsis_horiz->data; dpsis_vert_ptr = dpsis_vert->data;

        // upperleft corner
        sigma_u = dpsis_horiz_ptr[0]*du_ptr[1];
        sigma_v = dpsis_horiz_ptr[0]*dv_ptr[1];
        sigma_u += dpsis_vert_ptr[0]*du_ptr[stride];
        sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride];
        B1 = b1_ptr[0]+sigma_u;
        B2 = b2_ptr[0]+sigma_v;
        du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
        dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

        du_ptr++; dv_ptr++;
        A11_ptr++; A12_ptr++; A22_ptr++;
        b1_ptr++; b2_ptr++;
        dpsis_horiz_ptr++; dpsis_vert_ptr++;
        // count++;
        // middle of the first line
        for( i = 1; i < du->width - 1; i++)
        {
            sigma_u = dpsis_horiz_ptr[-1]*du_ptr[-1] + dpsis_horiz_ptr[0]*du_ptr[1];
            sigma_v = dpsis_horiz_ptr[-1]*dv_ptr[-1] + dpsis_horiz_ptr[0]*dv_ptr[1];
            sigma_u += dpsis_vert_ptr[0]*du_ptr[stride];
            sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride];

            B1 = b1_ptr[0]+sigma_u;
            B2 = b2_ptr[0]+sigma_v;
            du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
            dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

            du_ptr++; dv_ptr++;
            A11_ptr++; A12_ptr++; A22_ptr++;
            b1_ptr++; b2_ptr++;
            dpsis_horiz_ptr++; dpsis_vert_ptr++;
            //count++;
        }

        // rightupper corner
        sigma_u = dpsis_horiz_ptr[-1]*du_ptr[-1];
        sigma_v = dpsis_horiz_ptr[-1]*dv_ptr[-1];
        sigma_u += dpsis_vert_ptr[0]*du_ptr[stride];
        sigma_v += dpsis_vert_ptr[0]*dv_ptr[stride];
        B1 = b1_ptr[0]+sigma_u;
        B2 = b2_ptr[0]+sigma_v;
        du_ptr[0] += omega*( A11_ptr[0]*B1 + A12_ptr[0]*B2 - du_ptr[0] );
        dv_ptr[0] += omega*( A12_ptr[0]*B1 + A22_ptr[0]*B2 - dv_ptr[0] );

        //send the pointer to next line
        du_ptr+=new_line_incr; dv_ptr+=new_line_incr;
        b1_ptr+=new_line_incr; b2_ptr+=new_line_incr;
        dpsis_horiz_ptr += new_line_incr; dpsis_vert_ptr += new_line_incr;
        A11_ptr+=new_line_incr; A12_ptr+=new_line_incr; A22_ptr+=new_line_incr;


        //printf("first line offset: %ld ", du_ptr - du->data);

        for(j = 0; j < j_block_iter; ++j)
        {
            // First column
            for(int ee = 0; ee < 4; ++ee)
            {
                int offset = ee*stride;
                sigma_u = dpsis_horiz_ptr[offset]*du_ptr[1+offset];
                sigma_v = dpsis_horiz_ptr[offset]*dv_ptr[1+offset];
                sigma_u += dpsis_vert_ptr[offset]*du_ptr[stride+offset] + dpsis_vert_ptr[stride_+offset] * du_ptr[stride_+offset];
                sigma_v += dpsis_vert_ptr[offset]*dv_ptr[stride+offset] + dpsis_vert_ptr[stride_+offset] * dv_ptr[stride_+offset];
                B1 = b1_ptr[offset]+sigma_u;
                B2 = b2_ptr[offset]+sigma_v;
                du_ptr[offset] += omega*( A11_ptr[offset]*B1 + A12_ptr[offset]*B2 - du_ptr[offset] );
                dv_ptr[offset] += omega*( A12_ptr[offset]*B1 + A22_ptr[offset]*B2 - dv_ptr[offset] );
            }

            du_ptr++; dv_ptr++;
            A11_ptr++; A12_ptr++; A22_ptr++;
            b1_ptr++; b2_ptr++;
            dpsis_horiz_ptr++; dpsis_vert_ptr++;


            /*     16 17 18 19
             * 20 |00 01 02 03| 28
             * 21 |04 05 06 07| 29
             * 22 |08 09 10 11| 30
             * 23 |12 13 14 15| 31
             *     24 25 26 27
             *
             * 00 -> 01 04 -> 02 05 08 -> 03 06 09 12 -> 07 10 13 -> 11 14 -> 15
             * Maximum 4-way parallelism
             * */

            for( i = 0; i < i_block_iter; ++i)
            {
                // Load everything, so memory aliasing won't be a problem
                // vertical displacement
                float dp_vert_20 = dpsis_vert_ptr[-1];
                float dp_vert_00 = dpsis_vert_ptr[0];
                float dp_vert_01 = dpsis_vert_ptr[1];
                float dp_vert_02 = dpsis_vert_ptr[2];
                float dp_vert_03 = dpsis_vert_ptr[3];
                float dp_vert_28 = dpsis_vert_ptr[4];

                float dp_vert_21 = dpsis_vert_ptr[stride - 1];
                float dp_vert_04 = dpsis_vert_ptr[stride];
                float dp_vert_05 = dpsis_vert_ptr[stride + 1];
                float dp_vert_06 = dpsis_vert_ptr[stride + 2];
                float dp_vert_07 = dpsis_vert_ptr[stride + 3];
                float dp_vert_29 = dpsis_vert_ptr[stride + 4];

                float dp_vert_22 = dpsis_vert_ptr[stride*2 - 1];
                float dp_vert_08 = dpsis_vert_ptr[stride*2];
                float dp_vert_09 = dpsis_vert_ptr[stride*2 + 1];
                float dp_vert_10 = dpsis_vert_ptr[stride*2 + 2];
                float dp_vert_11 = dpsis_vert_ptr[stride*2 + 3];
                float dp_vert_30 = dpsis_vert_ptr[stride*2 + 4];

                float dp_vert_23 = dpsis_vert_ptr[stride*3 - 1];
                float dp_vert_12 = dpsis_vert_ptr[stride*3];
                float dp_vert_13 = dpsis_vert_ptr[stride*3 + 1];
                float dp_vert_14 = dpsis_vert_ptr[stride*3 + 2];
                float dp_vert_15 = dpsis_vert_ptr[stride*3 + 3];
                float dp_vert_31 = dpsis_vert_ptr[stride*3 + 4];
                // Up
                float dp_vert_16 = dpsis_vert_ptr[stride_];
                float dp_vert_17 = dpsis_vert_ptr[stride_ + 1];
                float dp_vert_18 = dpsis_vert_ptr[stride_ + 2];
                float dp_vert_19 = dpsis_vert_ptr[stride_ + 3];
                // Bottom
                float dp_vert_24 = dpsis_vert_ptr[stride*4];
                float dp_vert_25 = dpsis_vert_ptr[stride*4 + 1];
                float dp_vert_26 = dpsis_vert_ptr[stride*4 + 2];
                float dp_vert_27 = dpsis_vert_ptr[stride*4 + 3];

                // Horizontal displacement
                float dp_horiz_20 = dpsis_horiz_ptr[-1];
                float dp_horiz_00 = dpsis_horiz_ptr[0];
                float dp_horiz_01 = dpsis_horiz_ptr[1];
                float dp_horiz_02 = dpsis_horiz_ptr[2];
                float dp_horiz_03 = dpsis_horiz_ptr[3];
                float dp_horiz_28 = dpsis_horiz_ptr[4];

                float dp_horiz_21 = dpsis_horiz_ptr[stride - 1];
                float dp_horiz_04 = dpsis_horiz_ptr[stride];
                float dp_horiz_05 = dpsis_horiz_ptr[stride + 1];
                float dp_horiz_06 = dpsis_horiz_ptr[stride + 2];
                float dp_horiz_07 = dpsis_horiz_ptr[stride + 3];
                float dp_horiz_29 = dpsis_horiz_ptr[stride + 4];

                float dp_horiz_22 = dpsis_horiz_ptr[stride*2 - 1];
                float dp_horiz_08 = dpsis_horiz_ptr[stride*2];
                float dp_horiz_09 = dpsis_horiz_ptr[stride*2 + 1];
                float dp_horiz_10 = dpsis_horiz_ptr[stride*2 + 2];
                float dp_horiz_11 = dpsis_horiz_ptr[stride*2 + 3];
                float dp_horiz_30 = dpsis_horiz_ptr[stride*2 + 4];

                float dp_horiz_23 = dpsis_horiz_ptr[stride*3 - 1];
                float dp_horiz_12 = dpsis_horiz_ptr[stride*3];
                float dp_horiz_13 = dpsis_horiz_ptr[stride*3 + 1];
                float dp_horiz_14 = dpsis_horiz_ptr[stride*3 + 2];
                float dp_horiz_15 = dpsis_horiz_ptr[stride*3 + 3];
                float dp_horiz_31 = dpsis_horiz_ptr[stride*3 + 4];
                // Up
                float dp_horiz_16 = dpsis_horiz_ptr[stride_];
                float dp_horiz_17 = dpsis_horiz_ptr[stride_ + 1];
                float dp_horiz_18 = dpsis_horiz_ptr[stride_ + 2];
                float dp_horiz_19 = dpsis_horiz_ptr[stride_ + 3];
                // Bottom
                float dp_horiz_24 = dpsis_horiz_ptr[stride*4];
                float dp_horiz_25 = dpsis_horiz_ptr[stride*4 + 1];
                float dp_horiz_26 = dpsis_horiz_ptr[stride*4 + 2];
                float dp_horiz_27 = dpsis_horiz_ptr[stride*4 + 3];

                // du
                float du_20 = du_ptr[-1];
                float du_00 = du_ptr[0];
                float du_01 = du_ptr[1];
                float du_02 = du_ptr[2];
                float du_03 = du_ptr[3];
                float du_28 = du_ptr[4];

                float du_21 = du_ptr[stride - 1];
                float du_04 = du_ptr[stride];
                float du_05 = du_ptr[stride + 1];
                float du_06 = du_ptr[stride + 2];
                float du_07 = du_ptr[stride + 3];
                float du_29 = du_ptr[stride + 4];

                float du_22 = du_ptr[stride*2 - 1];
                float du_08 = du_ptr[stride*2];
                float du_09 = du_ptr[stride*2 + 1];
                float du_10 = du_ptr[stride*2 + 2];
                float du_11 = du_ptr[stride*2 + 3];
                float du_30 = du_ptr[stride*2 + 4];

                float du_23 = du_ptr[stride*3 - 1];
                float du_12 = du_ptr[stride*3];
                float du_13 = du_ptr[stride*3 + 1];
                float du_14 = du_ptr[stride*3 + 2];
                float du_15 = du_ptr[stride*3 + 3];
                float du_31 = du_ptr[stride*3 + 4];
                // Up
                float du_16 = du_ptr[stride_];
                float du_17 = du_ptr[stride_ + 1];
                float du_18 = du_ptr[stride_ + 2];
                float du_19 = du_ptr[stride_ + 3];
                // Bottom
                float du_24 = du_ptr[stride*4];
                float du_25 = du_ptr[stride*4 + 1];
                float du_26 = du_ptr[stride*4 + 2];
                float du_27 = du_ptr[stride*4 + 3];

                // dv
                float dv_20 = dv_ptr[-1];
                float dv_00 = dv_ptr[0];
                float dv_01 = dv_ptr[1];
                float dv_02 = dv_ptr[2];
                float dv_03 = dv_ptr[3];
                float dv_28 = dv_ptr[4];

                float dv_21 = dv_ptr[stride - 1];
                float dv_04 = dv_ptr[stride];
                float dv_05 = dv_ptr[stride + 1];
                float dv_06 = dv_ptr[stride + 2];
                float dv_07 = dv_ptr[stride + 3];
                float dv_29 = dv_ptr[stride + 4];

                float dv_22 = dv_ptr[stride*2 - 1];
                float dv_08 = dv_ptr[stride*2];
                float dv_09 = dv_ptr[stride*2 + 1];
                float dv_10 = dv_ptr[stride*2 + 2];
                float dv_11 = dv_ptr[stride*2 + 3];
                float dv_30 = dv_ptr[stride*2 + 4];

                float dv_23 = dv_ptr[stride*3 - 1];
                float dv_12 = dv_ptr[stride*3];
                float dv_13 = dv_ptr[stride*3 + 1];
                float dv_14 = dv_ptr[stride*3 + 2];
                float dv_15 = dv_ptr[stride*3 + 3];
                float dv_31 = dv_ptr[stride*3 + 4];
                // Up
                float dv_16 = dv_ptr[stride_];
                float dv_17 = dv_ptr[stride_ + 1];
                float dv_18 = dv_ptr[stride_ + 2];
                float dv_19 = dv_ptr[stride_ + 3];
                // Bottom
                float dv_24 = dv_ptr[stride*4];
                float dv_25 = dv_ptr[stride*4 + 1];
                float dv_26 = dv_ptr[stride*4 + 2];
                float dv_27 = dv_ptr[stride*4 + 3];


                // Load A11, A12, A22, B1, B2
                //b1
                float b1_00 = b1_ptr[0];
                float b1_01 = b1_ptr[1];
                float b1_02 = b1_ptr[2];
                float b1_03 = b1_ptr[3];

                float b1_04 = b1_ptr[stride];
                float b1_05 = b1_ptr[stride + 1];
                float b1_06 = b1_ptr[stride + 2];
                float b1_07 = b1_ptr[stride + 3];

                float b1_08 = b1_ptr[stride*2];
                float b1_09 = b1_ptr[stride*2 + 1];
                float b1_10 = b1_ptr[stride*2 + 2];
                float b1_11 = b1_ptr[stride*2 + 3];

                float b1_12 = b1_ptr[stride*3];
                float b1_13 = b1_ptr[stride*3 + 1];
                float b1_14 = b1_ptr[stride*3 + 2];
                float b1_15 = b1_ptr[stride*3 + 3];

                //b2
                float b2_00 = b2_ptr[0];
                float b2_01 = b2_ptr[1];
                float b2_02 = b2_ptr[2];
                float b2_03 = b2_ptr[3];

                float b2_04 = b2_ptr[stride];
                float b2_05 = b2_ptr[stride + 1];
                float b2_06 = b2_ptr[stride + 2];
                float b2_07 = b2_ptr[stride + 3];

                float b2_08 = b2_ptr[stride*2];
                float b2_09 = b2_ptr[stride*2 + 1];
                float b2_10 = b2_ptr[stride*2 + 2];
                float b2_11 = b2_ptr[stride*2 + 3];

                float b2_12 = b2_ptr[stride*3];
                float b2_13 = b2_ptr[stride*3 + 1];
                float b2_14 = b2_ptr[stride*3 + 2];
                float b2_15 = b2_ptr[stride*3 + 3];

                //A11
                float A11_00 = A11_ptr[0];
                float A11_01 = A11_ptr[1];
                float A11_02 = A11_ptr[2];
                float A11_03 = A11_ptr[3];

                float A11_04 = A11_ptr[stride];
                float A11_05 = A11_ptr[stride + 1];
                float A11_06 = A11_ptr[stride + 2];
                float A11_07 = A11_ptr[stride + 3];

                float A11_08 = A11_ptr[stride*2];
                float A11_09 = A11_ptr[stride*2 + 1];
                float A11_10 = A11_ptr[stride*2 + 2];
                float A11_11 = A11_ptr[stride*2 + 3];

                float A11_12 = A11_ptr[stride*3];
                float A11_13 = A11_ptr[stride*3 + 1];
                float A11_14 = A11_ptr[stride*3 + 2];
                float A11_15 = A11_ptr[stride*3 + 3];

                // A12
                float A12_00 = A12_ptr[0];
                float A12_01 = A12_ptr[1];
                float A12_02 = A12_ptr[2];
                float A12_03 = A12_ptr[3];

                float A12_04 = A12_ptr[stride];
                float A12_05 = A12_ptr[stride + 1];
                float A12_06 = A12_ptr[stride + 2];
                float A12_07 = A12_ptr[stride + 3];

                float A12_08 = A12_ptr[stride*2];
                float A12_09 = A12_ptr[stride*2 + 1];
                float A12_10 = A12_ptr[stride*2 + 2];
                float A12_11 = A12_ptr[stride*2 + 3];

                float A12_12 = A12_ptr[stride*3];
                float A12_13 = A12_ptr[stride*3 + 1];
                float A12_14 = A12_ptr[stride*3 + 2];
                float A12_15 = A12_ptr[stride*3 + 3];

                // A22
                float A22_00 = A22_ptr[0];
                float A22_01 = A22_ptr[1];
                float A22_02 = A22_ptr[2];
                float A22_03 = A22_ptr[3];

                float A22_04 = A22_ptr[stride];
                float A22_05 = A22_ptr[stride + 1];
                float A22_06 = A22_ptr[stride + 2];
                float A22_07 = A22_ptr[stride + 3];

                float A22_08 = A22_ptr[stride*2];
                float A22_09 = A22_ptr[stride*2 + 1];
                float A22_10 = A22_ptr[stride*2 + 2];
                float A22_11 = A22_ptr[stride*2 + 3];

                float A22_12 = A22_ptr[stride*3];
                float A22_13 = A22_ptr[stride*3 + 1];
                float A22_14 = A22_ptr[stride*3 + 2];
                float A22_15 = A22_ptr[stride*3 + 3];

                // 0
                {
                    sigma_u = dp_horiz_20 * du_20 + dp_horiz_00 * du_01 + dp_vert_16 * du_16 + dp_vert_00 * du_04;
                    sigma_v = dp_horiz_20 * dv_20 + dp_horiz_00 * dv_01 + dp_vert_16 * dv_16 + dp_vert_00 * dv_04;

                    B1 = b1_00 + sigma_u;
                    B2 = b2_00 + sigma_v;

                    du_00 += omega*( A11_00 * B1 + A12_00 * B2 - du_00 );
                    dv_00 += omega*( A12_00 * B1 + A22_00 * B2 - dv_00 );
                }


                // 2-way parallelism: 1, 4
                {
                    // 1
                    sigma_u = dp_horiz_00 * du_00 + dp_horiz_01 * du_02 + dp_vert_17 * du_17 + dp_vert_01 * du_05;
                    sigma_v = dp_horiz_00 * dv_00 + dp_horiz_01 * dv_02 + dp_vert_17 * dv_17 + dp_vert_01 * dv_05;

                    B1 = b1_01 + sigma_u;
                    B2 = b2_01 + sigma_v;

                    du_01 += omega*( A11_01 * B1 + A12_01 * B2 - du_01 );
                    dv_01 += omega*( A12_01 * B1 + A22_01 * B2 - dv_01 );

                    // 4
                    sigma_u = dp_horiz_21 * du_21 + dp_horiz_04 * du_05 + dp_vert_00 * du_00 + dp_vert_04 * du_08;
                    sigma_v = dp_horiz_21 * dv_21 + dp_horiz_04 * dv_05 + dp_vert_00 * dv_00 + dp_vert_04 * dv_08;

                    B1 = b1_04 + sigma_u;
                    B2 = b2_04 + sigma_v;

                    du_04 += omega*( A11_04 * B1 + A12_04 * B2 - du_04 );
                    dv_04 += omega*( A12_04 * B1 + A22_04 * B2 - dv_04 );
                }

                // 3-way parallelism: 2, 5, 8
                {
                    // 2
                    sigma_u = dp_horiz_01 * du_01 + dp_horiz_02 * du_03 + dp_vert_18 * du_18 + dp_vert_02 * du_06;
                    sigma_v = dp_horiz_01 * dv_01 + dp_horiz_02 * dv_03 + dp_vert_18 * dv_18 + dp_vert_02 * dv_06;

                    B1 = b1_02 + sigma_u;
                    B2 = b2_02 + sigma_v;

                    du_02 += omega*( A11_02 * B1 + A12_02 * B2 - du_02 );
                    dv_02 += omega*( A12_02 * B1 + A22_02 * B2 - dv_02 );

                    // 5
                    sigma_u = dp_horiz_04 * du_04 + dp_horiz_05 * du_06 + dp_vert_01 * du_01 + dp_vert_05 * du_09;
                    sigma_v = dp_horiz_04 * dv_04 + dp_horiz_05 * dv_06 + dp_vert_01 * dv_01 + dp_vert_05 * dv_09;

                    B1 = b1_05 + sigma_u;
                    B2 = b2_05 + sigma_v;

                    du_05 += omega*( A11_05 * B1 + A12_05 * B2 - du_05 );
                    dv_05 += omega*( A12_05 * B1 + A22_05 * B2 - dv_05 );

                    // 8
                    sigma_u = dp_horiz_22 * du_22 + dp_horiz_08 * du_09 + dp_vert_04 * du_04 + dp_vert_08 * du_12;
                    sigma_v = dp_horiz_22 * dv_22 + dp_horiz_08 * dv_09 + dp_vert_04 * dv_04 + dp_vert_08 * dv_12;

                    B1 = b1_08 + sigma_u;
                    B2 = b2_08 + sigma_v;

                    du_08 += omega*( A11_08 * B1 + A12_08 * B2 - du_08 );
                    dv_08 += omega*( A12_08 * B1 + A22_08 * B2 - dv_08 );

                }

                // 4-way parallelism: 3, 6, 9, 12
                {
                    // 3
                    sigma_u = dp_horiz_02 * du_02 + dp_horiz_03 * du_28 + dp_vert_19 * du_19 + dp_vert_03 * du_07;
                    sigma_v = dp_horiz_02 * dv_02 + dp_horiz_03 * dv_28 + dp_vert_19 * dv_19 + dp_vert_03 * dv_07;

                    B1 = b1_03 + sigma_u;
                    B2 = b2_03 + sigma_v;

                    du_03 += omega*( A11_03 * B1 + A12_03 * B2 - du_03 );
                    dv_03 += omega*( A12_03 * B1 + A22_03 * B2 - dv_03 );

                    // 6
                    sigma_u = dp_horiz_05 * du_05 + dp_horiz_06 * du_07 + dp_vert_02 * du_02 + dp_vert_06 * du_10;
                    sigma_v = dp_horiz_05 * dv_05 + dp_horiz_06 * dv_07 + dp_vert_02 * dv_02 + dp_vert_06 * dv_10;

                    B1 = b1_06 + sigma_u;
                    B2 = b2_06 + sigma_v;

                    du_06 += omega*( A11_06 * B1 + A12_06 * B2 - du_06 );
                    dv_06 += omega*( A12_06 * B1 + A22_06 * B2 - dv_06 );

                    // 9
                    sigma_u = dp_horiz_08 * du_08 + dp_horiz_09 * du_10 + dp_vert_05 * du_05 + dp_vert_09 * du_13;
                    sigma_v = dp_horiz_08 * dv_08 + dp_horiz_09 * dv_10 + dp_vert_05 * dv_05 + dp_vert_09 * dv_13;

                    B1 = b1_09 + sigma_u;
                    B2 = b2_09 + sigma_v;

                    du_09 += omega*( A11_09 * B1 + A12_09 * B2 - du_09 );
                    dv_09 += omega*( A12_09 * B1 + A22_09 * B2 - dv_09 );

                    // 12
                    sigma_u = dp_horiz_23 * du_23 + dp_horiz_12 * du_13 + dp_vert_08 * du_08 + dp_vert_12 * du_24;
                    sigma_v = dp_horiz_23 * dv_23 + dp_horiz_12 * dv_13 + dp_vert_08 * dv_08 + dp_vert_12 * dv_24;

                    B1 = b1_12 + sigma_u;
                    B2 = b2_12 + sigma_v;

                    du_12 += omega*( A11_12 * B1 + A12_12 * B2 - du_12 );
                    dv_12 += omega*( A12_12 * B1 + A22_12 * B2 - dv_12 );
                }

                // 3-way parallelism: 7, 10, 13
                {
                    // 7
                    sigma_u = dp_horiz_06 * du_06 + dp_horiz_07 * du_29 + dp_vert_03 * du_03 + dp_vert_07 * du_11;
                    sigma_v = dp_horiz_06 * dv_06 + dp_horiz_07 * dv_29 + dp_vert_03 * dv_03 + dp_vert_07 * dv_11;

                    B1 = b1_07 + sigma_u;
                    B2 = b2_07 + sigma_v;

                    du_07 += omega*( A11_07 * B1 + A12_07 * B2 - du_07 );
                    dv_07 += omega*( A12_07 * B1 + A22_07 * B2 - dv_07 );

                    // 10
                    sigma_u = dp_horiz_09 * du_09 + dp_horiz_10 * du_11 + dp_vert_06 * du_06 + dp_vert_10 * du_14;
                    sigma_v = dp_horiz_09 * dv_09 + dp_horiz_10 * dv_11 + dp_vert_06 * dv_06 + dp_vert_10 * dv_14;

                    B1 = b1_10 + sigma_u;
                    B2 = b2_10 + sigma_v;

                    du_10 += omega*( A11_10 * B1 + A12_10 * B2 - du_10 );
                    dv_10 += omega*( A12_10 * B1 + A22_10 * B2 - dv_10 );

                    // 13
                    sigma_u = dp_horiz_12 * du_12 + dp_horiz_13 * du_14 + dp_vert_09 * du_09 + dp_vert_13 * du_25;
                    sigma_v = dp_horiz_12 * dv_12 + dp_horiz_13 * dv_14 + dp_vert_09 * dv_09 + dp_vert_13 * dv_25;

                    B1 = b1_13 + sigma_u;
                    B2 = b2_13 + sigma_v;

                    du_13 += omega*( A11_13 * B1 + A12_13 * B2 - du_13 );
                    dv_13 += omega*( A12_13 * B1 + A22_13 * B2 - dv_13 );
                }

                // 2-way parallelism: 11, 14
                {
                    // 11
                    sigma_u = dp_horiz_10 * du_10 + dp_horiz_11 * du_30 + dp_vert_07 * du_07 + dp_vert_11 * du_15;
                    sigma_v = dp_horiz_10 * dv_10 + dp_horiz_11 * dv_30 + dp_vert_07 * dv_07 + dp_vert_11 * dv_15;

                    B1 = b1_11 + sigma_u;
                    B2 = b2_11 + sigma_v;

                    du_11 += omega*( A11_11 * B1 + A12_11 * B2 - du_11 );
                    dv_11 += omega*( A12_11 * B1 + A22_11 * B2 - dv_11 );

                    // 14
                    sigma_u = dp_horiz_13 * du_13 + dp_horiz_14 * du_15 + dp_vert_10 * du_10 + dp_vert_14 * du_26;
                    sigma_v = dp_horiz_13 * dv_13 + dp_horiz_14 * dv_15 + dp_vert_10 * dv_10 + dp_vert_14 * dv_26;

                    B1 = b1_14 + sigma_u;
                    B2 = b2_14 + sigma_v;

                    du_14 += omega*( A11_14 * B1 + A12_14 * B2 - du_14 );
                    dv_14 += omega*( A12_14 * B1 + A22_14 * B2 - dv_14 );
                }

                // 15
                {
                    sigma_u = dp_horiz_14 * du_14 + dp_horiz_15 * du_31 + dp_vert_11 * du_11 + dp_vert_15 * du_27;
                    sigma_v = dp_horiz_14 * dv_14 + dp_horiz_15 * dv_31 + dp_vert_11 * dv_11 + dp_vert_15 * dv_27;

                    B1 = b1_15 + sigma_u;
                    B2 = b2_15 + sigma_v;

                    du_15 += omega*( A11_15 * B1 + A12_15 * B2 - du_15 );
                    dv_15 += omega*( A12_15 * B1 + A22_15 * B2 - dv_15 );

                }

                // Write back to memory
                du_ptr[0] = du_00;
                du_ptr[1] = du_01;
                du_ptr[2] = du_02;
                du_ptr[3] = du_03;

                du_ptr[stride] = du_04;
                du_ptr[stride + 1] = du_05;
                du_ptr[stride + 2] = du_06;
                du_ptr[stride + 3] = du_07;

                du_ptr[stride*2] = du_08;
                du_ptr[stride*2 + 1] = du_09;
                du_ptr[stride*2 + 2] = du_10;
                du_ptr[stride*2 + 3] = du_11;

                du_ptr[stride*3] = du_12;
                du_ptr[stride*3 + 1] = du_13;
                du_ptr[stride*3 + 2] = du_14;
                du_ptr[stride*3 + 3] = du_15;

                dv_ptr[0] = dv_00;
                dv_ptr[1] = dv_01;
                dv_ptr[2] = dv_02;
                dv_ptr[3] = dv_03;

                dv_ptr[stride] = dv_04;
                dv_ptr[stride + 1] = dv_05;
                dv_ptr[stride + 2] = dv_06;
                dv_ptr[stride + 3] = dv_07;

                dv_ptr[stride*2] = dv_08;
                dv_ptr[stride*2 + 1] = dv_09;
                dv_ptr[stride*2 + 2] = dv_10;
                dv_ptr[stride*2 + 3] = dv_11;

                dv_ptr[stride*3] = dv_12;
                dv_ptr[stride*3 + 1] = dv_13;
                dv_ptr[stride*3 + 2] = dv_14;
                dv_ptr[stride*3 + 3] = dv_15;

                // Pass pointers to next block
                du_ptr+=4; dv_ptr+=4;
                A11_ptr+=4; A12_ptr+=4; A22_ptr+=4;
                b1_ptr+=4; b2_ptr+=4;
                dpsis_horiz_ptr+=4; dpsis_vert_ptr+=4;
            }

            //printf("block step offset: %ld ", du_ptr -du->data);
            // Do we need to pad the rest of columns
            for(int ec = 0; ec < extra_col; ++ec)
            {
                // There is dependency between rows. So no need to unroll that specifically. Or we can do it in a very
                // complex way...not worth it.
                for(int ee = 0; ee < 4; ++ee)
                {
                    int offset = ee*stride;
                    sigma_u = dpsis_horiz_ptr[-1+offset]*du_ptr[-1+offset] + dpsis_horiz_ptr[0+offset]*du_ptr[1+offset];
                    sigma_v = dpsis_horiz_ptr[-1+offset]*dv_ptr[-1+offset] + dpsis_horiz_ptr[0+offset]*dv_ptr[1+offset];
                    sigma_u += dpsis_vert_ptr[stride_+offset] * du_ptr[stride_+offset] + dpsis_vert_ptr[0+offset]*du_ptr[stride+offset];
                    sigma_v += dpsis_vert_ptr[stride_+offset] * dv_ptr[stride_+offset] + dpsis_vert_ptr[0+offset]*dv_ptr[stride+offset];
                    B1 = b1_ptr[offset]+sigma_u;
                    B2 = b2_ptr[offset]+sigma_v;
                    du_ptr[offset] += omega*( A11_ptr[offset]*B1 + A12_ptr[offset]*B2 - du_ptr[offset] );
                    dv_ptr[offset] += omega*( A12_ptr[offset]*B1 + A22_ptr[offset]*B2 - dv_ptr[offset] );
                }
                du_ptr++; dv_ptr++;
                A11_ptr++; A12_ptr++; A22_ptr++;
                b1_ptr++; b2_ptr++;
                dpsis_horiz_ptr++; dpsis_vert_ptr++;
            }

            // Last column
            for(int ee = 0; ee < 4; ++ee)
            {
                int offset = ee*stride;
                sigma_u = dpsis_horiz_ptr[-1+offset]*du_ptr[-1+offset];
                sigma_v = dpsis_horiz_ptr[-1+offset]*dv_ptr[-1+offset];
                sigma_u += dpsis_vert_ptr[offset]*du_ptr[stride+offset] + dpsis_vert_ptr[stride_+offset] * du_ptr[stride_+offset];
                sigma_v += dpsis_vert_ptr[offset]*dv_ptr[stride+offset] + dpsis_vert_ptr[stride_+offset] * dv_ptr[stride_+offset];
                B1 = b1_ptr[offset]+sigma_u;
                B2 = b2_ptr[offset]+sigma_v;
                du_ptr[offset] += omega*( A11_ptr[offset]*B1 + A12_ptr[offset]*B2 - du_ptr[offset] );
                dv_ptr[offset] += omega*( A12_ptr[offset]*B1 + A22_ptr[offset]*B2 - dv_ptr[offset] );
            }

            du_ptr+=block_line_incr; dv_ptr+=block_line_incr;
            A11_ptr+=block_line_incr; A12_ptr+=block_line_incr; A22_ptr+=block_line_incr;
            b1_ptr+=block_line_incr; b2_ptr+=block_line_incr;
            dpsis_horiz_ptr+=block_line_incr; dpsis_vert_ptr+=block_line_incr;

            //printf("block line offset: %ld ", du_ptr - du->data);
        }

        //printf("block total offset: %ld ", du_ptr - du->data);

        // Check the left rows
        for(int k = extra_row; k >= 0; --k) {
            // left column
            sigma_u = dpsis_horiz_ptr[0] * du_ptr[1];
            sigma_v = dpsis_horiz_ptr[0] * dv_ptr[1];
            sigma_u += dpsis_vert_ptr[stride_] * du_ptr[stride_];
            sigma_v += dpsis_vert_ptr[stride_] * dv_ptr[stride_];

            if (k > 0) {
                sigma_u += dpsis_vert_ptr[0] * du_ptr[stride];
                sigma_v += dpsis_vert_ptr[0] * dv_ptr[stride];
            }

            B1 = b1_ptr[0] + sigma_u;
            B2 = b2_ptr[0] + sigma_v;
            du_ptr[0] += omega * (A11_ptr[0] * B1 + A12_ptr[0] * B2 - du_ptr[0]);
            dv_ptr[0] += omega * (A12_ptr[0] * B1 + A22_ptr[0] * B2 - dv_ptr[0]);

            du_ptr++; dv_ptr++;
            A11_ptr++; A12_ptr++; A22_ptr++;
            b1_ptr++; b2_ptr++;
            dpsis_horiz_ptr++; dpsis_vert_ptr++;

            // middle of the first line
            for (i = 1; i < du->width - 1; ++i) {
                sigma_u = dpsis_horiz_ptr[-1] * du_ptr[-1] + dpsis_horiz_ptr[0] * du_ptr[1];
                sigma_v = dpsis_horiz_ptr[-1] * dv_ptr[-1] + dpsis_horiz_ptr[0] * dv_ptr[1];
                sigma_u += dpsis_vert_ptr[stride_] * du_ptr[stride_];
                sigma_v += dpsis_vert_ptr[stride_] * dv_ptr[stride_];

                if (k > 0) {
                    sigma_u += dpsis_vert_ptr[0] * du_ptr[stride];
                    sigma_v += dpsis_vert_ptr[0] * dv_ptr[stride];
                }

                B1 = b1_ptr[0] + sigma_u;
                B2 = b2_ptr[0] + sigma_v;
                du_ptr[0] += omega * (A11_ptr[0] * B1 + A12_ptr[0] * B2 - du_ptr[0]);
                dv_ptr[0] += omega * (A12_ptr[0] * B1 + A22_ptr[0] * B2 - dv_ptr[0]);

                du_ptr++; dv_ptr++;
                A11_ptr++; A12_ptr++; A22_ptr++;
                b1_ptr++; b2_ptr++;
                dpsis_horiz_ptr++; dpsis_vert_ptr++;
            }

            // right
            sigma_u = dpsis_horiz_ptr[-1] * du_ptr[-1];
            sigma_v = dpsis_horiz_ptr[-1] * dv_ptr[-1];
            sigma_u += dpsis_vert_ptr[stride_] * du_ptr[stride_];
            sigma_v += dpsis_vert_ptr[stride_] * dv_ptr[stride_];

            if (k > 0) {
                sigma_u += dpsis_vert_ptr[0] * du_ptr[stride];
                sigma_v += dpsis_vert_ptr[0] * dv_ptr[stride];
            }

            B1 = b1_ptr[0] + sigma_u;
            B2 = b2_ptr[0] + sigma_v;
            du_ptr[0] += omega * (A11_ptr[0] * B1 + A12_ptr[0] * B2 - du_ptr[0]);
            dv_ptr[0] += omega * (A12_ptr[0] * B1 + A22_ptr[0] * B2 - dv_ptr[0]);

            //send the pointer to next line
            du_ptr += new_line_incr;
            dv_ptr += new_line_incr;
            A11_ptr += new_line_incr;
            A12_ptr += new_line_incr;
            A22_ptr += new_line_incr;
            b1_ptr += new_line_incr;
            b2_ptr += new_line_incr;
            dpsis_horiz_ptr += new_line_incr;
            dpsis_vert_ptr += new_line_incr;
        }
        //printf("end offset: %ld\n", (du_ptr - du->data));
    }

    //printf("j block iter: %d, i block iter: %d\n", j_block_iter, i_block_iter);


    image_delete(A11); image_delete(A12); image_delete(A22);

}
