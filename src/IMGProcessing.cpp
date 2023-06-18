/*
 * Project:         DeepCPP
 *  Author:          John Doe <FreeHomeless@github.com> Copyright (c) 2023
 *  License:         MIT
 *  Description:     Image processing for the input layers of neural network using DeepCPP layers.
 *                   This files helps to reduce the amount of data that means less inputs in your 
 *                   layer. 
 */

#include "IMGProcessing.h"
#include "algebra.h"
#include <algorithm>
#include <cassert>
#include "typedefs.h"
#include <cmath>

#ifndef __linux__
#include <omp>
#endif

/*
 * Sums the elements of a matrix.
 *
 * @param mat Matrix # Input matrix.
 * @return double
 */
double DeepCPP::IMGProcessing::sumMatrix (const Matrix &mat) {
    static double acum = 0.0f;
    std::for_each(mat.begin(), mat.end(), [](Vector row){
        std::for_each(row.begin(), row.end(), [](double element) {
            acum += element;
        });
    });
    return acum;
}

/*
 * Using convolution technique, apply a mask over the original matrix. There are a collection of 
 * masks named MASK_{item}.
 *
 * @param origin Matrix* # Original matrix. The function returns a pointer to it but modifying the content.
 * @param mask Matrix # Mask that will be apply over the origin.
 * @return Matrix*
 */
Matrix* DeepCPP::IMGProcessing::applyMaskConvolution (Matrix* origin, const Matrix &mask) {
    static const size_t origin_dim[2] = dimension(*origin);
    static const size_t mask_dim[2] = dimension(mask);
#ifndef __linux__
#pragma omp parallel for
#endif
    for (int i=0; i<origin_dim[0]-mask_dim[0]; ++i){
        for (int j=0; j<origin_dim[1]-mask_dim[1]; ++j){
            for (int ii=i; ii<i + mask_dim[0]; ++ii){
                for (int jj=j; jj<j + mask_dim[1]; ++jj)
                    (*origin)[ii][jj] += (*origin)[ii][jj] * mask[ii-i][jj-j];           
            }
        }
    }
    return origin;
}

/*
 * This function reduces the entropy of a Matrix averaging the matrix into other matrix underscaled
 * by the dimension of the mask. If origin has 64x64 elements and the mask has 4x4, then the result
 * will be a matrix of 16x16 elements.
 *
 * @param origin Matrix* # Original Matrix.
 * @param mask Matrix # Mask that will be apply over the origin.
 * @return Matrix
 */
Matrix DeepCPP::IMGProcessing::applyMaskAverager (Matrix* origin, const Matrix &mask) {
    static const size_t origin_dim[2] = dimension(*origin);
    static const size_t mask_dim[2] = dimension(mask);

    Matrix retval((size_t) ceil(origin_dim[0] / mask_dim[0]), 
            Vector(ceil(origin_dim[1] / mask_dim[1])));
    static const size_t retval_dim[2] = dimension(retval);
#ifndef __linux__
#pragma omp parallel for
#endif
    for (int i=0; i<retval_dim[0]; ++i){
        for (int j=0; j<retval_dim[1]; ++j){
            Matrix m1(mask_dim[0], Vector(mask_dim[1]));

            for (int ii=retval_dim[0]*i; ii<retval_dim[0]*(i+1) && ii < origin_dim[0]; ++ii) {
                for (int jj=retval_dim[1]*j; jj<retval_dim[1]*(j+1) && jj < origin_dim[1]; ++jj) {
                    m1[ii - retval_dim[0]*i][jj - retval_dim[1]*jj] = (*origin)[ii][jj];
                }
            }
            retval[i][j] = DeepCPP::IMGProcessing::sumMatrix(dot(m1, mask));
        }
    }
    return retval;
}
