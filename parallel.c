#include <stdlib.h>
#include <omp.h>

#include "utils.h"
#include "parallel.h"

/*
 *  PHASE 1: compute the mean pixel value
 *  This code is buggy! Find the bug and speed it up.
 */
void mean_pixel_parallel(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols, double mean[NUM_CHANNELS])
{
    int row, col, ch;
    long count = num_cols*num_rows;
#pragma omp parallel
{
    #pragma omp for
        for(ch = 0; ch < NUM_CHANNELS; ch++)
            mean[ch] = 0;
    
            

    #pragma omp for collapse(2) reduction(+:mean[:NUM_CHANNELS]) //private(row, col, ch)
        for (row = 0; row < num_rows; row++)
        {
            for (col = 0; col < num_cols; col++)
            {
                mean[0] += img[row * num_cols + col][0];
                mean[1] += img[row * num_cols + col][1];
                mean[2] += img[row * num_cols + col][2];             
            }
        }


    #pragma omp for
    for(ch = 0; ch < NUM_CHANNELS; ch++)
            mean[ch] /= count;

}
}


/*
 *  PHASE 2: convert image to grayscale and record the max grayscale value along with the number of times it appears
 *  This code is NOT buggy, just sequential. Speed it up.
 */

void grayscale_parallel(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols, uint32_t grayscale_img[][NUM_CHANNELS], uint8_t *max_gray, uint32_t *max_count)
{
    int row, col, ch, gray_ch;
    *max_gray = 0;
    *max_count = 0;

    #pragma omp parallel 
    {
    uint8_t local_max_gray = 0;
    uint8_t local_max_count = 0;

        #pragma omp for collapse(2)
        for (row = 0; row < num_rows; row++)
        {
            for(col = 0; col < num_cols; col++)
            {
                //for(gray_ch = 0; gray_ch < NUM_CHANNELS; gray_ch++)
               // {
                    int indx = row * num_cols + col;

                    //0
                    grayscale_img[indx][0] = 0;

                    grayscale_img[indx][0] += img[indx][0];
                    grayscale_img[indx][0] += img[indx][1];
                    grayscale_img[indx][0] += img[indx][2];

                    grayscale_img[indx][0] /= NUM_CHANNELS;
                    
                    if (grayscale_img[indx][0] == local_max_gray)
                    {
                        local_max_count++;
                    }
                    else if (grayscale_img[indx][0] > local_max_gray)
                    {
                        local_max_gray = grayscale_img[indx][0];
                        local_max_count = 1;
                    }

                    //1
                    grayscale_img[indx][1] = 0;

                    grayscale_img[indx][1] += img[indx][0];
                    grayscale_img[indx][1] += img[indx][1];
                    grayscale_img[indx][1] += img[indx][2];

                    grayscale_img[indx][1] /= NUM_CHANNELS;
                    
                    if (grayscale_img[indx][1] == local_max_gray)
                    {
                        local_max_count++;
                    }
                    else if (grayscale_img[indx][1] > local_max_gray)
                    {
                        local_max_gray = grayscale_img[indx][1];
                        local_max_count = 1;
                    }

                //2
                    grayscale_img[indx][2] = 0;

                    grayscale_img[indx][2] += img[indx][0];
                    grayscale_img[indx][2] += img[indx][1];
                    grayscale_img[indx][2] += img[indx][2];

                    grayscale_img[indx][2] /= NUM_CHANNELS;
                    
                    if (grayscale_img[indx][2] == local_max_gray)
                    {
                        local_max_count++;
                    }
                    else if (grayscale_img[indx][2] > local_max_gray)
                    {
                        local_max_gray = grayscale_img[indx][2];
                        local_max_count = 1;
                    }
               // }
            }
        }

        #pragma omp critical
        {
            if(local_max_gray > *max_gray){
                *max_gray = local_max_gray;
                *max_count = local_max_count;
            }
            else if(local_max_gray == *max_gray){
                *max_count += local_max_count;
            }
        }
    }
}

/*
 *  PHASE 3: perform convolution on image
 *  This code is NOT buggy, just sequential. Speed it up.
 */
 
void convolution_parallel(const uint8_t padded_img[][NUM_CHANNELS], int num_rows, int num_cols, const uint32_t kernel[], int kernel_size, uint32_t convolved_img[][NUM_CHANNELS])
{
    int row, col, ch, kernel_row, kernel_col;
    int kernel_norm, i;
    int conv_rows, conv_cols, indx;

    // compute kernel normalization factor
    kernel_norm = 0;

    #pragma omp parallel for reduction(+:kernel_norm)
    for (i = 0; i < kernel_size * kernel_size; i++)
    {
        kernel_norm += kernel[i];
    }

    // compute dimensions of convolved image
    conv_rows = num_rows - kernel_size + 1;
    conv_cols = num_cols - kernel_size + 1;

    // perform convolution

    #pragma omp parallel for collapse(3) private(indx)
    for (ch = 0; ch < NUM_CHANNELS; ch++)
    {
        for (row = 0; row < conv_rows; row++)
        {
            for (col = 0; col < conv_cols; col++)
            {
                indx = row * conv_cols + col;
                convolved_img[indx][ch] = 0;
                for (kernel_col = 0; kernel_col < kernel_size; kernel_col++)
                {
                    for (kernel_row = 0; kernel_row < kernel_size; kernel_row++)
                    {
                        convolved_img[indx][ch] += padded_img[(row + kernel_row) * num_cols + col + kernel_col][ch] * kernel[kernel_row * kernel_size + kernel_col];
                    }
                }
                convolved_img[indx][ch] /= kernel_norm;
            }
        }
    }
}

