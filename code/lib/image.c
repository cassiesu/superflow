#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <malloc.h>
#include <math.h>
#include "image.h"


#ifdef __APPLE__
	#include <malloc/malloc.h>

void *memalign(size_t blocksize, size_t bytes) {
	void *result=0;
	posix_memalign(&result, blocksize, bytes);
	return result;
}

#else
	#include <malloc.h>
#endif

/********** Create/Delete **********/

/* allocate a new image of size width x height */
image_t *image_new(int width, int height)
{
	image_t *image = (image_t*) malloc(sizeof(image_t));
	if(image == NULL)
		{
			fprintf(stderr, "Error: image_new() - not enough memory !\n");
			exit(1);
		}
	image->width = width;
	image->height = height;
	image->stride = ((width + 7) / 8) * 8;
	image->data = (float*) memalign(32, image->stride*height*sizeof(float));
	if(image->data == NULL)
		{
			fprintf(stderr, "Error: image_new() - not enough memory !\n");
			exit(1);
		}
	return image;
}

/* allocate continues memory for image data */
float *image_data_new(int width, int height, int num) {
	float *pointer;
	int stride = ((width + 7) / 8) * 8;
	pointer = (float*) memalign(32, stride * height * sizeof(float) * num);
	if (pointer == NULL) {
		fprintf(stderr, "Error: image_daat_new() - not enough memory !\n");
		exit(1);
	} else {
		return pointer;
	}
}

/* allocate new image using given continues data memory */
image_t *image_with_data_new(int width, int height, float *pointer, 
		int offset) {
	image_t *image = (image_t*) malloc(sizeof(image_t));
	if (image == NULL) {
		fprintf(stderr, 
			"Error: image_with_data_new() - not enough memory !\n");
		exit(1);
	} else {
		image->width = width;
		image->height = height;
		image->stride = ((width + 7) / 8) * 8;
		image->data = pointer + offset * image->stride * height;
		if (image->data == NULL) {
			fprintf(stderr, 
				"Error: image_with_data_new() - offset out of range!\n");
			exit(1);
		} else {
			return image;
		}
	}
}

/* free memory of an image with continues data memory */
void image_with_data_delete(image_t *image) {
	if (image == NULL) {
		fprintf(stderr, 
			"Warning: image_with_data_delete() - image not allocated!\n");
	} else {
		free(image);
	}
}

/* free memory of continues data memory */
void image_data_delete(float *pointer) {
	if (pointer == NULL) {
		fprintf(stderr, 
			"Warning: image_data_delete() - data not allocated!\n");
	} else {
		free(pointer);
	}
}

/* allocate a new image and copy the content from src */
image_t *image_cpy(const image_t *src)
{
	image_t *dst = image_new(src->width, src->height);
	memcpy(dst->data, src->data, src->stride*src->height*sizeof(float));
	return dst;
}

// reset image width and height
void image_set(image_t *image, int width, int height) {
	if (image == NULL) {
		fprintf(stderr, "Error: image_set() - empty memory !\n");
		exit(1);
	} else {
		image->width = width;
		image->height = height;
		image->stride = ((width + 7) / 8 ) * 8;
	}
}

/* set all pixels values to zeros */
void image_erase(image_t *image)
{
	memset(image->data, 0, image->stride*image->height*sizeof(float));
}


/* multiply an image by a scalar */
void image_mul_scalar(image_t *image, float scalar)
{
	int i;
	for( i=0 ; i<image->stride*image->height ; i++)
		image->data[i] *= scalar;
}

/* free memory of an image */
void image_delete(image_t *image)
{
	if(image == NULL)
		{
			//fprintf(stderr, "Warning: Delete image --> Ignore action (image not allocated)\n");
		}
	else
		{
			free(image->data);
			free(image);
		}
}


/* allocate a new color image of size width x height */
color_image_t *color_image_new(int width, int height)
{
	size_t stride_channel = width*height*sizeof(float);
	char *buffer = (char*) malloc(sizeof(color_image_t) + 3*stride_channel);
	if(buffer == NULL)
		{
			fprintf(stderr, "Error: color_image_new() - not enough memory !\n");
			exit(1);
		}
	color_image_t *image = (color_image_t*) buffer;
	image->width = width;
	image->height = height;
	image->c1 = (float*) (buffer + sizeof(color_image_t));
	image->c2 = (float*) (buffer + sizeof(color_image_t) + stride_channel);
	image->c3 = (float*) (buffer + sizeof(color_image_t) + 2*stride_channel);
	return image;
}

/* allocate a new color image and copy the content from src */
color_image_t *color_image_cpy(const color_image_t *src)
{
	color_image_t *dst = color_image_new(src->width, src->height);
	memcpy(dst->c1, src->c1, 3*src->width*src->height*sizeof(float));
	return dst;
}

// reset image width and height
void color_image_set(color_image_t *image, int width, int height) {
	if(image == NULL) {
		fprintf(stderr, "Error: color_image_set() - empty memory !\n");
		exit(1);
	} else {
		image->width = width;
		image->height = height;
	}
}

/* set all pixels values to zeros */
void color_image_erase(color_image_t *image)
{
		memset(image->c1, 0, 3*image->width*image->height*sizeof(float));
}

/* free memory of a color image */
void color_image_delete(color_image_t *image)
{
	if(image) 
		{
			free(image); // the image is allocated such that the data is stored just after the pointer
		}
}

/* reallocate the memory of an image to fit the new width height */
void resize_if_needed_newsize(image_t *im, int w, int h)
{
	if(im->width != w || im->height != h)
		{
			im->width = w;
			im->height = h;
			im->stride = ((w + 7) / 8) * 8;
			float *data = (float *) memalign(32, im->stride*h*sizeof(float));
			if(data == NULL)
	{
		fprintf(stderr, "Error: resize_if_needed_newsize() - not enough memory !\n");
		exit(1);
	}
			free(im->data);
			im->data = data;
		}
}


/************ Resizing *********/

/* resize an image to a new size (assumes a difference only in width) */
void image_resize_horiz(image_t *dst, const image_t *src)
{
	int i;
	float real_scale = ((float) src->width-1) / ((float) dst->width-1);
	for(i = 0; i < dst->height; i++)
		{
			int j;
			for(j = 0; j < dst->width; j++)
				{
		float dx;
		int x;
		x = floor((float) j * real_scale);
		dx = j * real_scale - x; 
		if(x >= (src->width - 1))
						{
				dst->data[i * dst->stride + j] = 
		src->data[i * src->stride + src->width - 1]; 
						}
		else
						{
				dst->data[i * dst->stride + j] = 
		(1.0f - dx) * src->data[i * src->stride + x    ] + 
		(       dx) * src->data[i * src->stride + x + 1];
						}
				}
		}
}

/* resize a color image to a new size (assumes a difference only in width) */
void color_image_resize_horiz(color_image_t *dst, const color_image_t *src)
{
	int i;
	float real_scale = ((float) src->width-1) / ((float) dst->width-1);
	for(i = 0; i < dst->height; i++)
		{
			int j;
			for(j = 0; j < dst->width; j++)
				{
		int x;
		float dx;
		x = floor((float) j * real_scale);
		dx = j * real_scale - x; 
		if(x >= (src->width - 1))
						{
				dst->c1[i * dst->width + j] = 
		src->c1[i * src->width + src->width - 1]; 
				dst->c2[i * dst->width + j] = 
		src->c2[i * src->width + src->width - 1]; 
				dst->c3[i * dst->width + j] = 
		src->c3[i * src->width + src->width - 1]; 
						}
		else
						{
				dst->c1[i * dst->width + j] = 
		(1.0f - dx) * src->c1[i * src->width + x    ] + 
		(       dx) * src->c1[i * src->width + x + 1];
				dst->c2[i * dst->width + j] = 
		(1.0f - dx) * src->c2[i * src->width + x    ] + 
		(       dx) * src->c2[i * src->width + x + 1];
				dst->c3[i * dst->width + j] = 
		(1.0f - dx) * src->c3[i * src->width + x    ] + 
		(       dx) * src->c3[i * src->width + x + 1];
						}
				}
		}
}

/* resize an image to a new size (assumes a difference only in height) */
void image_resize_vert(image_t *dst, const image_t *src)
{
	int i;
	float real_scale = ((float) src->height-1) / ((float) dst->height-1);
	for(i = 0; i < dst->width; i++)
		{
			int j;
			for(j = 0; j < dst->height; j++)
				{
		int y;
		float dy;
		y = floor((float) j * real_scale);
		dy = j * real_scale - y;
		if(y >= (src->height - 1))
						{
				dst->data[j * dst->stride + i] =
		src->data[i + (src->height - 1) * src->stride]; 
						}
		else
						{
				dst->data[j * dst->stride + i] =
		(1.0f - dy) * src->data[i + (y    ) * src->stride] + 
		(       dy) * src->data[i + (y + 1) * src->stride];
						}
				}
		}
}

/* resize a color image to a new size (assumes a difference only in height) */
void color_image_resize_vert(color_image_t *dst, const color_image_t *src)
{
	int i;
	float real_scale = ((float) src->height) / ((float) dst->height);
	for(i = 0; i < dst->width; i++)
		{
			int j;
			for(j = 0; j < dst->height; j++)
				{
		int y;
		float dy;
		y = floor((float) j * real_scale);
		dy = j * real_scale - y;
		if(y >= (src->height - 1))
						{
				dst->c1[j * dst->width + i] =
		src->c1[i + (src->height - 1) * src->width]; 
				dst->c2[j * dst->width + i] =
		src->c2[i + (src->height - 1) * src->width]; 
				dst->c3[j * dst->width + i] =
		src->c3[i + (src->height - 1) * src->width]; 
						}
		else
						{
				dst->c1[j * dst->width + i] =
		(1.0f - dy) * src->c1[i +  y      * src->width] + 
		(       dy) * src->c1[i + (y + 1) * src->width];
							dst->c2[j * dst->width + i] =
		(1.0f - dy) * src->c2[i +  y      * src->width] + 
		(       dy) * src->c2[i + (y + 1) * src->width];
							dst->c3[j * dst->width + i] =
		(1.0f - dy) * src->c3[i +  y      * src->width] + 
		(       dy) * src->c3[i + (y + 1) * src->width];
						}
				}
		}
}

/* return a resize version of the image with bilinear interpolation */
image_t *image_resize_bilinear(const image_t *src, float scale)
{
	int width = src->width, height = src->height;
	int newwidth = (int) (1.5f + (width-1) / scale); // 0.5f for rounding instead of flooring, and the remaining comes from scale = (dst-1)/(src-1)
	int newheight = (int) (1.5f + (height-1) / scale);
	image_t *dst = image_new(newwidth,newheight);
	if(height*newwidth < width*newheight)
		{
			image_t *tmp = image_new(newwidth,height);
			image_resize_horiz(tmp,src);
			image_resize_vert(dst,tmp);
			image_delete(tmp);
		}
	else
		{
			image_t *tmp = image_new(width,newheight);
			image_resize_vert(tmp,src);
			image_resize_horiz(dst,tmp);
			image_delete(tmp);
		}
	return dst;
}

/* resize an image with bilinear interpolation to fit the new weidht, height ; reallocation is done if necessary */
void image_resize_bilinear_newsize(image_t *dst, const image_t *src, int new_width, int new_height)
{
	resize_if_needed_newsize(dst,new_width,new_height);
	if(new_width < new_height)
		{
			image_t *tmp = image_new(new_width,src->height);
			image_resize_horiz(tmp,src);
			image_resize_vert(dst,tmp);
			image_delete(tmp);
		}
	else
		{
			image_t *tmp = image_new(src->width,new_height);
			image_resize_vert(tmp,src);
			image_resize_horiz(dst,tmp); 
			image_delete(tmp);
		}
}

/* resize a color image  with bilinear interpolation */
color_image_t *color_image_resize_bilinear(const color_image_t *src, float scale)
{
	int width = src->width, height = src->height;
	int newwidth = (int) (1.5f + (width-1) / scale); // 0.5f for rounding instead of flooring, and the remaining comes from scale = (dst-1)/(src-1)
	int newheight = (int) (1.5f + (height-1) / scale);
	color_image_t *dst = color_image_new(newwidth,newheight);
	if(height*newwidth < width*newheight)
		{
			color_image_t *tmp = color_image_new(newwidth,height);
			color_image_resize_horiz(tmp,src);
			color_image_resize_vert(dst,tmp);
			color_image_delete(tmp);
		}
	else
		{
			color_image_t *tmp = color_image_new(width,newheight);
			color_image_resize_vert(tmp,src);
			color_image_resize_horiz(dst,tmp);
			color_image_delete(tmp);
		}
	return dst;
}

/************ Convolution ******/

/* return half coefficient of a gaussian filter
Details:
- return a float* containing the coefficient from middle to border of the filter, so starting by 0,
- it so contains half of the coefficient.
- sigma is the standard deviation.
- filter_order is an output where the size of the output array is stored */
float *gaussian_filter(float sigma, int *filter_order)
{
	if(sigma == 0.0f)
		{
			fprintf(stderr, "gaussian_filter() error: sigma is zeros\n");
			exit(1);
		}
	if(!filter_order)
		{
			fprintf(stderr, "gaussian_filter() error: filter_order is null\n");
			exit(1);
		}
	// computer the filter order as 1 + 2* floor(3*sigma)
	*filter_order = floor(3*sigma); 
	if ( *filter_order == 0 )
		*filter_order = 1; 
	// compute coefficients
	float *data = malloc(sizeof(float) * (2*(*filter_order)+1));
	if(data == NULL )
		{
			fprintf(stderr, "gaussian_filter() error: not enough memory\n");
			exit(1);
		}
	float alpha = 1.0f/(2.0f*sigma*sigma), sum = 0.0f;
	int i;
	for(i=-(*filter_order) ; i<=*filter_order ; i++)
		{
			data[i+(*filter_order)] = exp(-i*i*alpha);
			sum += data[i+(*filter_order)];
		}
	for(i=-(*filter_order) ; i<=*filter_order ; i++)
		{
			data[i+(*filter_order)] /= sum;
		}
	// fill the output
	float *data2 = malloc(sizeof(float)*(*filter_order+1));
	if(data2 == NULL )
		{
			fprintf(stderr, "gaussian_filter() error: not enough memory\n");
			exit(1);
		}
	memcpy(data2, &data[*filter_order], sizeof(float)*(*filter_order)+sizeof(float));
	free(data);
	return data2;
}

/* given half of the coef, compute the full coefficients and the accumulated coefficients */
static void convolve_extract_coeffs(int order, const float *half_coeffs, float *coeffs, float *coeffs_accu, int even)
{
	int i;
	float accu = 0.0;
	if(even)
		{
			for(i = 0 ; i <= order; i++) 
	{
		coeffs[order - i] = coeffs[order + i] = half_coeffs[i];
				}
			for(i = 0 ; i <= order; i++)
				{
		accu += coeffs[i];
		coeffs_accu[2 * order - i] = coeffs_accu[i] = accu;
				}
		}
	else
		{
			for(i = 0; i <= order; i++)
				{
		coeffs[order - i] = +half_coeffs[i];
		coeffs[order + i] = -half_coeffs[i];
				}
			for(i = 0 ; i <= order; i++)
				{
		accu += coeffs[i];
		coeffs_accu[i] = accu;
		coeffs_accu[2 * order - i]= -accu;
				}
		}
}

/* create a convolution structure with a given order, half_coeffs, symmetric or anti-symmetric according to even parameter */
convolution_t *convolution_new(int order, const float *half_coeffs, int even)
{
	convolution_t *conv = (convolution_t *) malloc(sizeof(convolution_t));
	if(conv == NULL)
		{
			fprintf(stderr, "Error: convolution_new() - not enough memory !\n");
			exit(1);
		}
	conv->order = order;
	conv->coeffs = (float *) malloc((2 * order + 1) * sizeof(float));
	if(conv->coeffs == NULL)
		{
			fprintf(stderr, "Error: convolution_new() - not enough memory !\n");
			free(conv);
			exit(1);
		}
	conv->coeffs_accu = (float *) malloc((2 * order + 1) * sizeof(float));
	if(conv->coeffs_accu == NULL)
		{
			fprintf(stderr, "Error: convolution_new() - not enough memory !\n");
			free(conv->coeffs);
			free(conv);
			exit(1);
		}
	convolve_extract_coeffs(order, half_coeffs, conv->coeffs,conv->coeffs_accu, even);
	return conv;
}

/* perform an horizontal convolution of an image */
void convolve_horiz(image_t *dest, const image_t *src, const convolution_t *conv)
{
	float *in = src->data;
	float * out = dest->data;
	int i, j, ii;
	float *o = out;
	int i0 = -conv->order;
	int i1 = +conv->order;
	float *coeff = conv->coeffs + conv->order;
	float *coeff_accu = conv->coeffs_accu + conv->order;
	for(j = 0; j < src->height; j++)
		{
			const float *al = in + j * src->stride;
			const float *f0 = coeff + i0;
			float sum;
			for(i = 0; i < -i0; i++)
				{
		sum=coeff_accu[-i - 1] * al[0];
		for(ii = i1 + i; ii >= 0; ii--) 
			{
				sum += coeff[ii - i] * al[ii];
						}
		*o++ = sum;
				}
			for(; i < src->width - i1; i++)
				{
		sum = 0;
		for(ii = i1 - i0; ii >= 0; ii--) 
			{
				sum += f0[ii] * al[ii];
						}
		al++;
		*o++ = sum;
				}
			for(; i < src->width; i++)
				{
		sum = coeff_accu[src->width - i] * al[src->width - i0 - 1 - i];
		for(ii = src->width - i0 - 1 - i; ii >= 0; ii--) 
			{
				sum += f0[ii] * al[ii];
						}
		al++;
		*o++ = sum;
				}
			for(i = 0; i < src->stride - src->width; i++) 
	{
		o++;
				}
		}
}

/* perform a vertical convolution of an image */
void convolve_vert(image_t *dest, const image_t *src, const convolution_t *conv)
{
	float *in = src->data;
	float *out = dest->data;
	int i0 = -conv->order;
	int i1 = +conv->order;
	float *coeff = conv->coeffs + conv->order;
	float *coeff_accu = conv->coeffs_accu + conv->order;
	int i, j, ii;
	float *o = out;
	const float *alast = in + src->stride * (src->height - 1);
	const float *f0 = coeff + i0;
	for(i = 0; i < -i0; i++)
		{
			float fa = coeff_accu[-i - 1];
			const float *al = in + i * src->stride;
			for(j = 0; j < src->width; j++)
				{
		float sum = fa * in[j];
		for(ii = -i; ii <= i1; ii++) 
			{
				sum += coeff[ii] * al[j + ii * src->stride];
						}
		*o++ = sum;
				}
			for(j = 0; j < src->stride - src->width; j++) 
	{
		o++;
				}
		}
	for(; i < src->height - i1; i++)
		{
			const float *al = in + (i + i0) * src->stride;
			for(j = 0; j < src->width; j++)
	{
		float sum = 0;
		const float *al2 = al;
		for(ii = 0; ii <= i1 - i0; ii++)
						{
				sum += f0[ii] * al2[0];
				al2 += src->stride;
						}
		*o++ = sum;
		al++;
				}
			for(j = 0; j < src->stride - src->width; j++) 
	{
		o++;
				}
		}
	for(;i < src->height; i++)
		{
			float fa = coeff_accu[src->height - i];
			const float *al = in + i * src->stride;
			for(j = 0; j < src->width; j++)
				{
		float sum = fa * alast[j];
		for(ii = i0; ii <= src->height - 1 - i; ii++) 
			{
				sum += coeff[ii] * al[j + ii * src->stride];
						}
		*o++ = sum;
				}
			for(j = 0; j < src->stride - src->width; j++) 
	{
		o++;
				}
		}
}

/* free memory of a convolution structure */
void convolution_delete(convolution_t *conv)
{
	if(conv)
		{
			free(conv->coeffs);
			free(conv->coeffs_accu);
			free(conv);
		}
}

/* perform horizontal and/or vertical convolution to a color image */
void color_image_convolve_hv(color_image_t *dst, const color_image_t *src, const convolution_t *horiz_conv, const convolution_t *vert_conv)
{
	int width = src->width, height = src->height;
	// separate channels of images
	image_t src_red = {width,height,width,src->c1}, src_green = {width,height,width,src->c2}, src_blue = {width,height,width,src->c3}, 
					dst_red = {width,height,width,dst->c1}, dst_green = {width,height,width,dst->c2}, dst_blue = {width,height,width,dst->c3};
	// horizontal and vertical
	if(horiz_conv != NULL && vert_conv != NULL)
		{
			float *tmp_data = malloc(sizeof(float)*width*height);
			if(tmp_data == NULL)
	{
		fprintf(stderr,"error color_image_convolve_hv(): not enough memory\n");
		exit(1);
	}  
			image_t tmp = {width,height,width,tmp_data};   
			// perform convolution for each channel
			convolve_horiz(&tmp,&src_red,horiz_conv); 
			convolve_vert(&dst_red,&tmp,vert_conv); 
			convolve_horiz(&tmp,&src_green,horiz_conv);
			convolve_vert(&dst_green,&tmp,vert_conv); 
			convolve_horiz(&tmp,&src_blue,horiz_conv); 
			convolve_vert(&dst_blue,&tmp,vert_conv);
			free(tmp_data);
		}
	// only horizontal
	else if(horiz_conv != NULL && vert_conv == NULL)
		{
			convolve_horiz(&dst_red,&src_red,horiz_conv);
			convolve_horiz(&dst_green,&src_green,horiz_conv);
			convolve_horiz(&dst_blue,&src_blue,horiz_conv);
		}
	// only vertical
	else if(vert_conv != NULL && horiz_conv == NULL)
		{
			convolve_vert(&dst_red,&src_red,vert_conv);
			convolve_vert(&dst_green,&src_green,vert_conv);
			convolve_vert(&dst_blue,&src_blue,vert_conv);
		}
}

/************ Pyramid **********/

/* create new color image pyramid structures */
color_image_pyramid_t* color_image_pyramid_new()
{
	color_image_pyramid_t* pyr = (color_image_pyramid_t*) malloc(sizeof(color_image_pyramid_t));
	if(pyr == NULL)
		{
			fprintf(stderr,"Error in color_image_pyramid_new(): not enough memory\n");
			exit(1);
		}
	pyr->min_size = -1;
	pyr->scale_factor = -1.0f;
	pyr->size = -1;
	pyr->images = NULL;
	return pyr;
}

/* set the size of the color image pyramid structures (reallocate the array of pointers to images) */
void color_image_pyramid_set_size(color_image_pyramid_t* pyr, int size)
{
	if(size<0)
		{
			fprintf(stderr,"Error in color_image_pyramid_set_size(): size is negative\n");
			exit(1);
		}
	if(pyr->images == NULL)
		{
			pyr->images = (color_image_t**) malloc(sizeof(color_image_t*)*size);
		}
	else
		{
			pyr->images = (color_image_t**) realloc(pyr->images,sizeof(color_image_t*)*size);
		}
	if(pyr->images == NULL)
		{
			fprintf(stderr,"Error in color_image_pyramid_set_size(): not enough memory\n");
			exit(1);      
		}
	pyr->size = size;
}

/* create a pyramid of color images using a given scale factor, stopping when one dimension reach min_size and with applying a gaussian smoothing of standard deviation spyr (no smoothing if 0) */
color_image_pyramid_t *color_image_pyramid_create(const color_image_t *src, float scale_factor, int min_size, float spyr)
{
	int nb_max_scale = 200;
	// allocate structure
	color_image_pyramid_t *pyramid = color_image_pyramid_new();
	pyramid->min_size = min_size;
	pyramid->scale_factor = scale_factor;
	convolution_t *conv = NULL;
	if(spyr>0.0f)
		{
			int fsize;
			float *filter_coef = gaussian_filter(spyr, &fsize);
			conv = convolution_new(fsize, filter_coef, 1);
			free(filter_coef);
		}
	color_image_pyramid_set_size(pyramid, nb_max_scale);
	pyramid->images[0] = color_image_cpy(src);
	int i;
	for( i=1 ; i<nb_max_scale ; i++)
		{
			int oldwidth = pyramid->images[i-1]->width, oldheight = pyramid->images[i-1]->height;
			int newwidth = (int) (1.5f + (oldwidth-1) / scale_factor);
			int newheight = (int) (1.5f + (oldheight-1) / scale_factor);
			if( newwidth <= min_size || newheight <= min_size)
	{
		color_image_pyramid_set_size(pyramid, i);
		break;
	}
			if(spyr>0.0f)
	{
		color_image_t* tmp = color_image_new(oldwidth, oldheight);
		color_image_convolve_hv(tmp,pyramid->images[i-1], conv, conv);
		pyramid->images[i]= color_image_resize_bilinear(tmp, scale_factor);
		color_image_delete(tmp);
	}
			else
	{
		pyramid->images[i] = color_image_resize_bilinear(pyramid->images[i-1], scale_factor);
	}
		}
	if(spyr>0.0f)
		{
			convolution_delete(conv);
		}
	return pyramid;
}

/* delete the structure of a pyramid of color images and all the color images in it*/
void color_image_pyramid_delete(color_image_pyramid_t *pyr)
{
	if(pyr==NULL)
		{
			return;
		}
	int i;
	for(i=0 ; i<pyr->size ; i++)
		{
			color_image_delete(pyr->images[i]);
		}
	free(pyr->images);
	free(pyr);
}
