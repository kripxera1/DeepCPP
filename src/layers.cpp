/* 
 * File: src/layers.cpp
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Contains class definitions for different types of neural network layers.
 */

#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>

#include "layers.h"
#include "algebra.h"
#include "NNUtils.h"

// Layers
////////////////////////////////////////////////////////////////////////////////

Matrix Layer::getDelta(){
    return delta;
}

Matrix Layer::getInput(){
    return input;
}

////////////////////////////////////////////////////////////////////////////////

Linear::Linear(int input_size, int output_size) {

    W = Matrix(output_size, Vector(input_size, 0));
    b = Vector(output_size, 0);
    dW = Matrix(output_size, Vector(input_size, 0));
    db = Vector(output_size, 0);

    initWeightsBias(W, b);
}

Matrix Linear::forward(const Matrix& input_){
    input = input_;
    return sum(dot(W, input),b);
}

Matrix Linear::backward(const Matrix& prev_delta){

    // lineal entrada dZ
    dW = dot(prev_delta, T(input));
    db = rowsSum(prev_delta);
    delta = dot(T(W), prev_delta);

    return delta;
}


void Linear::initWeightsBias(Matrix & W, Vector & b){
    
    for(int i = 0; i < W.size();i++){
        b[i]=((double)rand()/RAND_MAX)-0.5;
        for(int j = 0; j < W[0].size(); j++){
            W[i][j]=((double)rand()/RAND_MAX)-0.5;
        }
    }
}

Vector Linear::getGradient() {

    int total_size = dW.size() * dW[0].size() + db.size();

    Vector gradient;
    gradient.reserve(total_size);

    for (const auto& row : dW) {
        for (const auto& elem : row) {
            gradient.push_back(elem);
        }
    }

    for (const auto& elem : db) {
        gradient.push_back(elem);
    }

    return gradient;
}

void Linear::scaleGradient(double scale) {
    for (auto& row : dW) {
        for (auto& elem : row) {
            elem *= scale;
        }
    }

    for (auto& elem : db) {
        elem *= scale;
    }
}

////////////////////////////////////////////////////////////////////////////////

LeakyRelu::LeakyRelu(double alpha) : alpha(alpha) {}

Matrix LeakyRelu::forward(const Matrix &input_) {
    input = input_;
    Matrix output;
    resize(output, input);

    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < output.size(); i++) {
        for (int j = 0; j < output[0].size(); j++) {
            output[i][j] = input[i][j] >= 0 ? input[i][j]:(alpha * input[i][j]);
        }
    }

    return output;
}

Matrix LeakyRelu::backward(const Matrix &prev_delta) {
    delta = hadamard(prev_delta, derivative(input));
    return delta;
}

Matrix LeakyRelu::derivative(const Matrix &input) {
    Matrix derivative;
    resize(derivative, input);
    
    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < derivative.size(); i++) {
        for (int j = 0; j < derivative[0].size(); j++) {
            derivative[i][j] = input[i][j] >= 0 ? 1.0 : alpha;
        }
    }

    return derivative;
}

////////////////////////////////////////////////////////////////////////////////

Matrix SoftMax::forward(const Matrix& input_) {
    input = input_;
    Matrix output;
    resize(output, input);

    #pragma omp parallel for
    for (int j = 0; j < input[0].size(); j++) {
        double max_val = -INFINITY;
        //Calculamos tamaño máximo
        for (int i = 0; i < input.size(); i++) {
            if (input[i][j] > max_val) {
                max_val = input[i][j];
            }
        }
        double sum = 0;
        for (int i = 0; i < input.size(); i++) {
            double exp_val = exp(input[i][j] - max_val);
            output[i][j] = exp_val;
            sum += exp_val;
        }
        for (int i = 0; i < input.size(); i++) {
            output[i][j] /= sum;
        }
    }

    return output;
}

Matrix SoftMax::backward(const Matrix& prev_delta) {
    delta = hadamard(prev_delta, derivative(input));
    return delta;
}

Matrix SoftMax::derivative(const Matrix &input) {
    Matrix derivative;
    resize(derivative, input);
    
    #pragma omp parallel for
    for (int i = 0; i < derivative.size(); i++) {
        
        double max_input = *std::max_element(input[i].begin(), input[i].end());

        double sum_exp = 0.0;
        for (int j = 0; j < derivative[0].size(); j++) {
            sum_exp += exp(input[i][j] - max_input);
        }
        
        for (int j = 0; j < derivative[0].size(); j++) {
            double softmax_val = exp(input[i][j] - max_input) / sum_exp;
            derivative[i][j] = softmax_val * (1.0 - softmax_val);
        }
    }

    return derivative;
}

////////////////////////////////////////////////////////////////////////////////

Matrix Relu::forward(const Matrix &input_) {
    input = input_;
    Matrix output;
    resize(output, input);

    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < output.size(); i++) {
        for (int j = 0; j < output[0].size(); j++) {
            output[i][j] = std::max(0.0, input[i][j]);
        }
    }

    return output;
}

Matrix Relu::backward(const Matrix &prev_delta) {
    delta = hadamard(prev_delta, derivative(input));
    return delta;
}

Matrix Relu::derivative(const Matrix &input_) {
    input = input_;
    Matrix derivative;
    resize(derivative, input);
    
    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < derivative.size(); i++) {
        for (int j = 0; j < derivative[0].size(); j++) {
            derivative[i][j] = input[i][j] > 0 ? 1.0 : 0.0;
        }
    }

    return derivative;
}

////////////////////////////////////////////////////////////////////////////////

Matrix Tanh::forward(const Matrix &input_) {
    input = input_;

    Matrix output;
    resize(output, input);

    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < output.size(); i++) {
        for (int j = 0; j < output[0].size(); j++) {
            output[i][j] = std::tanh(input[i][j]);
        }
    }

    return output;
}

Matrix Tanh::backward(const Matrix &prev_delta) {
    delta = hadamard(prev_delta, derivative(input));
    return delta;
}

Matrix Tanh::derivative(const Matrix &input) {
    Matrix derivative;
    resize(derivative, input);
    
    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < derivative.size(); i++) {
        for (int j = 0; j < derivative[0].size(); j++) {
            double tanh_val = tanh(input[i][j]);
            derivative[i][j] = 1.0 - tanh_val * tanh_val;
        }
    }

    return derivative;
}

////////////////////////////////////////////////////////////////////////////////

Matrix Sigmoid::forward(const Matrix &input_) {
    input = input_;
    Matrix output;
    resize(output, input);

    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < output.size(); i++) {
        for (int j = 0; j < output[0].size(); j++) {
            output[i][j] = 1 / (1 + std::exp(-input[i][j]));
        }
    }

    return output;
}

Matrix Sigmoid::backward(const Matrix &prev_delta) {
    delta = hadamard(prev_delta, derivative(input));
    return delta;
}


Matrix Sigmoid::derivative(const Matrix &input_) {
    input = input_;
    Matrix derivative;
    resize(derivative, input);
    
    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < derivative.size(); i++) {
        for (int j = 0; j < derivative[0].size(); j++) {
            double sigmoid_val = 1.0 / (1.0 + exp(-input[i][j]));
            derivative[i][j] = sigmoid_val * (1.0 - sigmoid_val);
        }
    }

    return derivative;
}

////////////////////////////////////////////////////////////////////////////////
Matrix NormalSampling::forward(const Matrix& input_) {
    input = input_;
    int n = input.size() / 2;

    resize(mu,input_);
    mu.resize(n);
    resize(log_var,input_);
    log_var.resize(n);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < input_[0].size(); ++j) {
            mu[i][j] = input_[i][j];
            log_var[i][j] = input_[i + n][j];
        }
    }

    Matrix output = Matrix(n,Vector(input[0].size()));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0, 1);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < input[0].size(); ++j) {
            double epsilon = dist(gen);
            output[i][j] = mu[i][j] + exp(0.5 * log_var[i][j]) * epsilon;
        }
    }

    return output;
}

Matrix NormalSampling::backward(const Matrix& prev_delta) {
    int n = prev_delta.size();
    delta.resize(prev_delta.size() * 2, Vector(prev_delta[0].size(), 0.0));

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < prev_delta[0].size(); ++j) {
            delta[i][j] = prev_delta[i][j];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < prev_delta[0].size(); ++j) {
            double grad = 0.5 * exp(log_var[i][j]) * pow(exp(0.5 * log_var[i][j]) * prev_delta[i][j], 2);
            delta[i + n][j] = grad;
        }
    }

    return delta;
}

Matrix Gelu::forward(const Matrix &input_) {
    input = input_;
    const double sqrt2_over_pi = std::sqrt(2.0 / M_PI);
    const double constant_0_044715 = 0.044715;
    Matrix output;
    resize(output,input);

    #pragma omp parallel for
    for (std::size_t i = 0; i < input.size(); ++i) {
        for (std::size_t j = 0; j < input[i].size(); ++j) {
            double x = input[i][j];
            double cdf = 0.5 * (1.0 + std::tanh(sqrt2_over_pi * (x + constant_0_044715 * x * x * x)));
            output[i][j] = x * cdf;
        }
    }

    return output;
}

Matrix Gelu::backward(const Matrix &prev_delta) {
    delta = hadamard(prev_delta, derivative(input));
    return delta;
}

Matrix Gelu::derivative(const Matrix &input_) {
    const double sqrt2_over_pi = std::sqrt(2.0 / M_PI);
    const double constant_0_044715 = 0.044715;
    Matrix derivative;
    resize(derivative,input);

   
    #pragma omp parallel for
    for (std::size_t i = 0; i < input.size(); ++i) {
        for (std::size_t j = 0; j < input[i].size(); ++j) {
            double x = input[i][j];
            double alpha = 1 + std::tanh(sqrt2_over_pi*(x + constant_0_044715 * x* x * x));
            double cdf = 0.5 * alpha;
            double pdf = 0.5 * alpha + 0.5 * (1.0 - cdf) * alpha
            * (sqrt2_over_pi * (1.0 + constant_0_044715 * 3.0 * x * x));
            derivative[i][j] = pdf;
        }
    }

    return derivative;
}

///////////////////////////////////////////////////////////////////////////////

// Revisar
Dropout::Dropout(double keep_probability_) 
: keep_probability(keep_probability_) {
    // Usa la hora actual como semilla para el generador de números aleatorios
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
}

Matrix Dropout::forward(const Matrix& input_) {
    input = input_;
    mask.resize(input.size(), std::vector<double>(input[0].size()));

    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < mask.size(); i++) {
        for (int j = 0; j < mask[0].size(); j++) {
            mask[i][j] = (distribution(generator) < keep_probability) ? 1.0 : 0.0;
            input[i][j] *= mask[i][j];  // No escalamos por keep_probability aquí
        }
    }

    return input;
}

Matrix Dropout::backward(const Matrix& prev_delta) {
    Matrix delta = prev_delta;

    for (int i = 0; i < delta.size(); i++) {
        for (int j = 0; j < delta[0].size(); j++) {
            delta[i][j] *= mask[i][j];
        }
    }

    return delta;
}