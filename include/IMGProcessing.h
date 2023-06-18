/* Project:         DeepCPP
 *  Author:          John Doe <FreeHomeless@github.com> Copyright (c) 2023
 *  License:         MIT
 *  Description:     Image processing for the input layers of neural network using DeepCPP layers.
 *                   This files helps to reduce the amount of data that means less inputs in your 
 *                   layer. */

#pragma once

#include "algebra.h"

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
namespace DeepCPP::IMGProcessing {
#else
    namespace DeepCPP {
        namespace IMGProcessing {
#endif

            double sumMatrix(const Matrix* mat);
            Matrix* applyMaskConvolution (Matrix* origin, const Matrix &mask);
            Matrix* applyMaskAverager (Matrix* origin, const Matrix &mask);

            const Matrix MASK_ZEROS ({
                    {0.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0}});

            const Matrix MASK_ONES ({
                    {1.0, 1.0, 1.0, 1.0},
                    {1.0, 1.0, 1.0, 1.0},
                    {1.0, 1.0, 1.0, 1.0},
                    {1.0, 1.0, 1.0, 1.0}});

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
        };
#else
    }
};
#endif
