/* 
 * File: src/optimizers.cpp
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Contains class definitions for different types of optimizers used in training the neural network.
 */

#include "optimizers.h"
#include <cmath>
#include <iostream>
#include <omp.h>

Adam::Adam(double beta1, double beta2, double epsilon)
    : beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}


void Adam::initialize(const Linear& layer) {
    OptimizationState state;
    state.mW = Matrix(layer.W.size(), Vector(layer.W[0].size(), 0.0));
    state.mb = Vector(layer.b.size(), 0.0);
    state.vW = Matrix(layer.W.size(), Vector(layer.W[0].size(), 0.0));
    state.vb = Vector(layer.b.size(), 0.0);

    optimization_states[&layer] = state;
}


void Adam::update(Linear& layer, double learn_rate, int batch_size) {
    ++t;

    OptimizationState& state = optimization_states[&layer];

    #pragma omp parallel for
    for (int i = 0; i < layer.b.size(); i++) {
        double grad_b = layer.db[i] / batch_size;
        state.mb[i] = beta1 * state.mb[i] + (1.0 - beta1) * grad_b;
        state.vb[i] = beta2 * state.vb[i] + (1.0 - beta2) * grad_b * grad_b;

        double m_hat_b = state.mb[i] / (1.0 - std::pow(beta1, t));
        double v_hat_b = state.vb[i] / (1.0 - std::pow(beta2, t));

        layer.b[i] -= learn_rate * m_hat_b / (std::sqrt(v_hat_b) + epsilon);
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < layer.W.size(); i++) {
        for (int j = 0; j < layer.W[0].size(); j++) {
            double grad_W = layer.dW[i][j] / batch_size;
            state.mW[i][j] = beta1 * state.mW[i][j] + (1.0 - beta1) * grad_W;
            state.vW[i][j] = beta2 * state.vW[i][j] + (1.0 - beta2) * grad_W * grad_W;

            double m_hat_W = state.mW[i][j] / (1.0 - std::pow(beta1, t));
            double v_hat_W = state.vW[i][j] / (1.0 - std::pow(beta2, t));

            layer.W[i][j] -= learn_rate * m_hat_W / (std::sqrt(v_hat_W) + epsilon);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void SGD::update(Linear& layer, double learn_rate, int batch_size) {
    for (int i = 0; i < layer.b.size(); i++) {
        layer.b[i] -= layer.db[i] * learn_rate / batch_size;
    }

    for (int i = 0; i < layer.W.size(); i++) {
        for (int j = 0; j < layer.W[0].size(); j++) {
            layer.W[i][j] -= layer.dW[i][j] * learn_rate / batch_size;
        }
    }
}