/* 
 * File: include/NNUtils.h
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Contains utility functions for neural networks.
 */


#ifndef NNUTILS_H
#define NNUTILS_H

#include <fstream>
#include <sstream>
#include <iterator>
#include <vector>
#include <cmath>
#include <memory>

#include "typedefs.h"
#include "algebra.h"
#include "losses.h"
#include "layers.h"
#include "optimizers.h"
#include "LRScheduler.h"


//Neural Network
//////////////////////////////////////////////////////////////////////////////

class NeuralNetwork {
public:

    vector<shared_ptr<Layer>> layers;

    shared_ptr<LossFunction> loss;

    shared_ptr<Optimizer> optimizer;

    NeuralNetwork(const vector<shared_ptr<Layer>>& layers_,
                  const shared_ptr<LossFunction>& loss_,
                  const shared_ptr<Optimizer>& optimizer_);

    Matrix forward(const Matrix& X);

    Matrix backward(const Matrix& output, const Matrix& expected_output);

    void update(double learn_rate, int batch_size);
};

vector<vector<int>> loadData(const char* file_name);

void loadBatch(const vector<vector<int>> &data,
               int batch_size,
               int it,
               Matrix &X,
               Matrix &Y);

void resize(Matrix& a, const Matrix& b);

vector<int> getPrediction(const Matrix &A);

double getAccuracy(const Matrix &A, const Matrix &Y);

void gradientClipping(NeuralNetwork &nn, double clip);

#endif // NNUTILS_H