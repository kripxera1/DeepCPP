/* 
 * File: src/LRScheduler.cpp
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Contains the learning rate scheduler class which allows for dynamic learning rates during training.
 */


#include "LRScheduler.h"
#include <cmath>

ConstantLearningRate::ConstantLearningRate(double learning_rate) 
    : learning_rate(learning_rate) {}

double ConstantLearningRate::getLearningRate(int epoch) {
    return learning_rate;
}

TriangularCyclicLR::TriangularCyclicLR(double base_learning_rate,
                                       double max_learning_rate,
                                       int step_size)
    : base_learning_rate(base_learning_rate),
      max_learning_rate(max_learning_rate),
      step_size(step_size) {}

double TriangularCyclicLR::getLearningRate(int epoch) {
    double cycle = std::floor(1 + epoch / (2 * step_size));
    double x = std::abs(epoch / step_size - 2 * cycle + 1);
    return base_learning_rate +
           (max_learning_rate - base_learning_rate) *
           std::max(0.0, (1 - x)) / std::pow(2, cycle - 1);
}


Triangular2CyclicLR::Triangular2CyclicLR(double base_lr,
                                         double max_lr,
                                         int step_size_up,
                                         int step_size_down)
    : base_lr(base_lr),
      max_lr(max_lr),
      step_size_up(step_size_up),
      step_size_down(step_size_down),
      cycle_iteration(0) {}

double Triangular2CyclicLR::getLearningRate(int iteration) {
    cycle_iteration++;
    int cycle = std::floor(1 + iteration / (2*(step_size_up + step_size_down)));
    int x = std::abs(iteration / (step_size_up + step_size_down) - 2*cycle + 1);
    double lr = base_lr + (max_lr - base_lr) *
           std::max(0.0, (double)(1 - x)) / std::pow(2, cycle - 1);

    return lr;
}