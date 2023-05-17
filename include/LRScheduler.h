/* 
 * File: include/LRScheduler.h
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Contains the learning rate scheduler class which allows for dynamic learning rates during training.
 */

#ifndef  lr_scheduleR_H
#define lr_scheduleR_H

class LearningRateScheduler {
public:
    virtual double getLearningRate(int epoch) = 0;
};


class ConstantLearningRate : public LearningRateScheduler {
public:
    ConstantLearningRate(double learning_rate);

    double getLearningRate(int epoch) override;

private:
    double learning_rate;
};


class TriangularCyclicLR : public LearningRateScheduler {
public:
    TriangularCyclicLR(double base_learning_rate,
                       double max_learning_rate,
                       int step_size);

    double getLearningRate(int epoch) override;

private:
    double base_learning_rate;
    double max_learning_rate;
    int step_size;
};


class Triangular2CyclicLR : public LearningRateScheduler {
public:
    Triangular2CyclicLR(double base_lr,
                        double max_lr,
                        int step_size_up,
                        int step_size_down);

    double getLearningRate(int iteration) override;

private:
    double base_lr;
    double max_lr;
    int step_size_up;
    int step_size_down;
    int cycle_iteration;
};

#endif // lr_scheduleR_H