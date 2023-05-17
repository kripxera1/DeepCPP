/* 
 * File: include/losses.h
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Contains declarations for different types of loss functions for training the neural network.
 */

#ifndef LOSSES_H
#define LOSSES_H
#include "typedefs.h"
// Loss Functions
//////////////////////////////////////////////////////////////////////////////

class LossFunction {
public:
    virtual double compute(const Matrix &A, const Matrix &Y) const = 0;
    virtual Matrix backward(const Matrix &A, const Matrix &Y) const = 0;
};

class CrossEntropy : public LossFunction {
public:
    double compute(const Matrix &A, const Matrix &Y) const override;
    Matrix backward(const Matrix &A, const Matrix &Y) const override;
};

class BinaryCrossEntropy : public LossFunction {
public:
    double compute(const Matrix &A, const Matrix &Y) const override;
    Matrix backward(const Matrix &A, const Matrix &Y) const override;
};

class MeanSquaredError : public LossFunction {
public:
    double compute(const Matrix &A, const Matrix &Y) const override;
    Matrix backward(const Matrix &A, const Matrix &Y) const override;
};


#endif // LOSSES_H