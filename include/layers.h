/* 
 * File: include/layers.h
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Contains class declarations for different types of neural network layers.
 */

#ifndef LAYERS_H
#define LAYERS_H

#include <random>
#include "typedefs.h"


// Layers
//////////////////////////////////////////////////////////////////////////////

class Layer {
protected:
    Matrix delta;
    Matrix input;

    double input_size;
    double output_size;
public:
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& delta) = 0;
    Matrix getDelta();
    Matrix getInput();
    virtual Vector getGradient() { return Vector(0); }
    virtual void scaleGradient(double scale) {}
};


class Linear : public Layer {
private:
    // weights and biases are initialized randomly between -0.5 and 0.5
    void initWeightsBias(Matrix & W, Vector & b);

public:
    // Weights and biases
    Matrix W;
    Vector b;

    // Gradient
    Matrix dW;
    Vector db;

    Linear(int input_size, int output_size);
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& prev_delta) override;
    Vector getGradient() override;
    void scaleGradient(double scale) override;
};


class Sigmoid : public Layer {
    Matrix derivative(const Matrix & input);
public:    
    Matrix forward(const Matrix &z) override;
    Matrix backward(const Matrix &prev_delta) override;    
};


class  Tanh : public Layer {
private:
    Matrix derivative(const Matrix & input);
public:
    Matrix forward(const Matrix &z) override;
    Matrix backward(const Matrix &prev_delta) override;
};


class Relu : public Layer {
private:
    Matrix derivative(const Matrix & input);
public:
    Matrix forward(const Matrix &z) override;
    Matrix backward(const Matrix &prev_delta) override;
};


class  LeakyRelu : public Layer {
private:
    Matrix derivative(const Matrix & input);
public:
    double alpha;
    LeakyRelu(double alpha);
    Matrix forward(const Matrix &z) override;
    Matrix backward(const Matrix &prev_delta) override;
};


class  SoftMax : public Layer {
private:
    Matrix derivative(const Matrix & input);
public:
    Matrix forward(const Matrix &z) override;
    Matrix backward(const Matrix &prev_delta) override;
};


class Gelu : public Layer {
private:
    Matrix derivative(const Matrix & input);
public:
    Matrix forward(const Matrix &z) override;
    Matrix backward(const Matrix &prev_delta) override;
};


class Dropout : public Layer {
private:
    Matrix mask;
    double keep_probability;
    std::default_random_engine generator;
public:
    Dropout(double keep_probability_);
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& prev_delta) override;
};


class NormalSampling : public Layer {
private:
    Matrix mu;
    Matrix log_var;
public:
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& prev_delta) override;
};


#endif // LAYERS_H