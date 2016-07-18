#ifndef __IMAGE_H_
#define __IMAGE_H_

#include <stdio.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MINMAX(a,b) MIN( MAX(a,0) , b-1 )

/********** STRUCTURES *********/

/* structure for 1-channel image */
typedef struct image_s
{
	int width;		/* Width of the image */
	int height;		/* Height of the image */
	int stride;		/* Width of the memory (width + paddind such that it is a multiple of 4) */
	float *data;		/* Image data */
} image_t;

/* structure for 3-channels image stored with one layer per color, it assumes that c2 = c1+width*height and c3 = c2+width*height. */
typedef struct color_image_s
{
		int width;			/* Width of the image */
		int height;			/* Height of the image */
		float *c1;			/* Color 1 */
		float *c2;			/* Color 2 */
		float *c3;			/* Color 3 */
} color_image_t;

/* structure for color image pyramid */
typedef struct color_image_pyramid_s 
{
	float scale_factor;          /* difference of scale between two levels */
	int min_size;                /* minimum size for width or height at the coarsest level */
	int size;                    /* number of levels in the pyramid */
	color_image_t **images;      /* list of images with images[0] the original one, images[size-1] the finest one */
} color_image_pyramid_t;

/* structure for convolutions */
typedef struct convolution_s
{
		int order;			/* Order of the convolution */
		float *coeffs;		/* Coefficients */
		float *coeffs_accu;		/* Accumulated coefficients */
} convolution_t;

/********** Create/Delete **********/

/* allocate a new image of size width x height */
image_t *image_new(int width, int height);

/* allocate continues memory for image data */
float *image_data_new(int width, int height, int num);

/* allocate new image using given continues data memory */
image_t *image_with_data_new(int width, int height, float *pointer, 
		int offset);

/* free memory of an image with continues data memory */
void image_with_data_delete(image_t *image);

/* free memory of continues data memory */
void image_data_delete(float *pointer);

/* allocate a new image and copy the content from src */
image_t *image_cpy(const image_t *src);

// reset image width and height
void image_set(image_t *image, int width, int height);

/* set all pixels values to zeros */
void image_erase(image_t *image);

/* free memory of an image */
void image_delete(image_t *image);

/* multiply an image by a scalar */
void image_mul_scalar(image_t *image, float scalar);

/* allocate a new color image of size width x height */
color_image_t *color_image_new(int width, int height);

/* allocate a new color image and copy the content from src */
color_image_t *color_image_cpy(const color_image_t *src);

// reset image width and height
void color_image_set(color_image_t *image, int width, int height);

/* set all pixels values to zeros */
void color_image_erase(color_image_t *image);

/* free memory of a color image */
void color_image_delete(color_image_t *image);

/* reallocate the memory of an image to fit the new width height */
void resize_if_needed_newsize(image_t *im, int w, int h);

/************ Resizing *********/

/* resize an image with bilinear interpolation */
image_t *image_resize_bilinear(const image_t *src, float scale);

/* resize an image with bilinear interpolation to fit the new weidht, height ; reallocation is done if necessary */
void image_resize_bilinear_newsize(image_t *dst, const image_t *src, int new_width, int new_height);

/* resize a color image  with bilinear interpolation */
color_image_t *color_image_resize_bilinear(const color_image_t *src, float scale);

/************ Convolution ******/

/* return half coefficient of a gaussian filter */
float *gaussian_filter(float sigma, int *fSize);

/* create a convolution structure with a given order, half_coeffs, symmetric or anti-symmetric according to even parameter */
convolution_t *convolution_new(int order, const float *half_coeffs, int even);

/* perform an horizontal convolution of an image */
void convolve_horiz(image_t *dest, const image_t *src, const convolution_t *conv);

/* perform a vertical convolution of an image */
void convolve_vert(image_t *dest, const image_t *src, const convolution_t *conv);

/* free memory of a convolution structure */
void convolution_delete(convolution_t *conv);

/* perform horizontal and/or vertical convolution to a color image */
void color_image_convolve_hv(color_image_t *dst, const color_image_t *src, const convolution_t *horiz_conv, const convolution_t *vert_conv);

/************ Pyramid **********/

/* create a pyramid of color images using a given scale factor, stopping when one dimension reach min_size and with applying a gaussian smoothing of standard deviation spyr (no smoothing if 0) */
color_image_pyramid_t *color_image_pyramid_create(const color_image_t *src, float scale_factor, int min_size, float spyr);

/* delete the structure of a pyramid of color images */
void color_image_pyramid_delete(color_image_pyramid_t *pyr);

#endif
