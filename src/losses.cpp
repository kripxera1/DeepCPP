/* 
 * File: src/losses.cpp
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Contains definitions for different types of loss functions for training the neural network.
 */

#include <cmath>
#include "losses.h"
#include "NNUtils.h"


// Loss functions
//////////////////////////////////////////////////////////////////////////////


double CrossEntropy::compute(const Matrix &A, const Matrix &Y) const {
    double loss = 0.0;
    int m = A[0].size();
    double epsilon = 1e-6;

    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A[0].size(); ++j) {
            loss -= Y[i][j] * log(std::max(A[i][j], epsilon)) +
              (1 - Y[i][j]) * log(std::max(1 - A[i][j], epsilon));
        }
    }

    return loss / m;
}

Matrix CrossEntropy::backward(const Matrix &A, const Matrix &Y) const {
    Matrix delta;
    resize(delta, A);
    double epsilon = 1e-9;

    for (int i = 0; i < delta.size(); ++i) {
        for (int j = 0; j < delta[0].size(); ++j) {
            if (Y[i][j] == 1.0) {
                delta[i][j] = -1.0 / (A[i][j] + epsilon);
            } else {
                delta[i][j] = 1.0 / (1.0 - A[i][j] + epsilon);
            }
        }
    }

    return delta;
}

double BinaryCrossEntropy::compute(const Matrix &A, const Matrix &Y) const {
    double loss = 0.0;
    int m = A[0].size();
    double epsilon = 1e-8;

    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A[0].size(); ++j) {
            loss -= Y[i][j] * log(A[i][j] + epsilon) + (1 - Y[i][j]) * log(1 - A[i][j] + epsilon);
        }
    }

    return loss / m;
}

Matrix BinaryCrossEntropy::backward(const Matrix &A, const Matrix &Y) const {
    Matrix dZ = A;
    double epsilon = 1e-8;

    for (int i = 0; i < dZ.size(); ++i) {
        for (int j = 0; j < dZ[0].size(); ++j) {
            dZ[i][j] = (A[i][j] - Y[i][j]) / ((A[i][j] + epsilon) * (1 - A[i][j] + epsilon));
        }
    }

    return dZ;
}

double MeanSquaredError::compute(const Matrix &A, const Matrix &Y) const {
    double loss = 0.0;
    int m = A[0].size();

    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A[0].size(); ++j) {
            double diff = A[i][j] - Y[i][j];
            loss += diff * diff;
        }
    }

    return loss / (2 * m);
}

Matrix MeanSquaredError::backward(const Matrix &A, const Matrix &Y) const {
    Matrix dZ = A;

    for (int i = 0; i < dZ.size(); ++i) {
        for (int j = 0; j < dZ[0].size(); ++j) {
            dZ[i][j] = (A[i][j] - Y[i][j]);
        }
    }

    return dZ;
}
