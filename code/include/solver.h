#ifndef __SOLVER_H_
#define __SOLVER_H_

#include <stdio.h>
#include <stdlib.h>

#include "image.h"

/* Perform niter iterations of the sor_coupled algorithm for a system of the form as described in the other file */
void sor_coupled(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, int niter, float omega);

void sor_coupled_slow_but_readable(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, const int iterations, const float omega);

void sor_coupled_slow_precompute_index(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, const int iterations, const float omega);
	
void sor_coupled_slow_scalar_replacement(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, const int iterations, const float omega);

void sor_coupled_blocked_1x4(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, int niter, float omega);

void sor_coupled_blocked_2x2(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, int niter, float omega);

void sor_coupled_blocked_2x2_vectorization(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, int niter, float omega);

void sor_coupled_blocked_4x4(image_t *du, image_t *dv, const image_t *a11, const image_t *a12, const image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, int niter, float omega);

#endif
