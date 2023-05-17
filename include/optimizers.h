/* 
 * File: include/optimizers.h
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Contains class declarations for different types of optimizers used in training the neural network.
 */

#include "layers.h"
#include "typedefs.h"
#include <map>

class Optimizer {
public:
    virtual void update(Linear& layer, double learn_rate, int batch_size) = 0;
    virtual void initialize(const Linear& layer) {};
};

class Adam : public Optimizer {
public:
    Adam(double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

    void initialize(const Linear& layer) override;
    void update(Linear& layer, double learn_rate, int batch_size) override;

private:
    struct OptimizationState {
        Matrix mW;
        Vector mb;
        Matrix vW;
        Vector vb;
    };

    double beta1;
    double beta2;
    double epsilon;
    int t;
    std::map<const Linear*, OptimizationState> optimization_states;
};

class SGD : public Optimizer {
public:
    void update(Linear& layer, double learn_rate, int batch_size) override;
};