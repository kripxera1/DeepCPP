/* 
 * File: src/NNUtils.cpp
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Contains utility functions for neural networks.
 */

#include <iostream>
#include "NNUtils.h"
#include "losses.h"
#include "layers.h"



// Neural network
//////////////////////////////////////////////////////////////////////////////


NeuralNetwork::NeuralNetwork(const std::vector<std::shared_ptr<Layer>>& layers_,
                             const std::shared_ptr<LossFunction>& loss_,
                             const std::shared_ptr<Optimizer>& optimizer_)
    : layers(layers_), loss(loss_), optimizer(optimizer_) {

    for (auto& layer : layers) {
        auto linear_layer = std::dynamic_pointer_cast<Linear>(layer);
        if (linear_layer) {
            optimizer->initialize(*linear_layer);
        }
    }
}



Matrix NeuralNetwork::forward(const Matrix& input) {

    Matrix current_input = input;

    for (auto& layer : layers) {
        current_input = layer->forward(current_input);
        
    }

    return current_input;
}



Matrix NeuralNetwork::backward(const Matrix & output, const Matrix& expected_output) {

    // If the combination of last layer and loss is softmax and crossentropy the
    // process is optimized using the difference between the output and expected
    // output, it then skips the last layer and loss.

    bool is_softmax_cross_entropy = 
        std::dynamic_pointer_cast<SoftMax>(layers.back()) && 
        std::dynamic_pointer_cast<CrossEntropy>(loss);
    
    Matrix delta = is_softmax_cross_entropy 
                   ? minusM(output,expected_output)
                   : loss->backward(output, expected_output);
    

    int start_layer = layers.size() - 1 - is_softmax_cross_entropy;

    for (int i = start_layer; i >= 0; i--) {
        delta = layers[i]->backward(delta);
    }

    return delta;
}



void NeuralNetwork::update(double learn_rate, int batch_size) {
    for (auto& layer : layers) {
        auto linear_layer = std::dynamic_pointer_cast<Linear>(layer);
        if (linear_layer) {
            optimizer->update(*linear_layer, learn_rate, batch_size);
        }
    }
}


// Data Loading
//////////////////////////////////////////////////////////////////////////////

vector<vector<int>> loadData(const char* file_name){

    ifstream file(file_name,ios::in);

    string line;
    vector<vector<int>> data;

    while (getline(file, line)) {
        istringstream is(line);
        data.push_back(vector<int>(istream_iterator<int>(is),
                       istream_iterator<int>()));
    }

    return data;
}



void loadBatch(const vector<vector<int>> &data,
               int batch_size,
               int it,
               Matrix &A,
               Matrix &one_hot) {

    int data_columns = data[0].size() - 1;
    A = Matrix(batch_size, Vector(data_columns, 0));
    one_hot = Matrix(batch_size, Vector(10, 0));

    int first = batch_size * it;
    int last = first + batch_size;

    for (int i = first, k = 0; i < last; i++, k++) {

        int label = data[i][0];
        one_hot[k][label] = 1;

        for (int j = 0; j < data_columns; j++) {
            A[k][j] = static_cast<double>(data[i][j + 1]) / 255;
        }
    }
    A = T(A);
    one_hot = T(one_hot);
}



vector<int> getPrediction(const Matrix & A){

    vector<int> v;
    for(int j = 0; j < A[0].size(); j++){
        int max_int;
        double max = -INFINITY;
        for(int i = 0; i < A.size(); i++)
            if(A[i][j]>max){
                max = A[i][j];
                max_int = i;                
            }
        v.push_back(max_int);
    }

    return v;
}



double getAccuracy(const Matrix & A, const Matrix & Y){

    double right = 0;
    for(int j = 0; j < A[0].size(); j++){
        int max_int;
        double max = -INFINITY;
        for(int i = 0; i < A.size(); i++)
            if(A[i][j]>max){
                max = A[i][j];
                max_int = i;
            }
        if(Y[max_int][j]==1)
            right+=1;
    }

    return right/Y[0].size();
}



void resize(Matrix& a, const Matrix& b) {
    a.resize(b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        a[i].resize(b[i].size());
    }
}



void gradientClipping(NeuralNetwork& network, double max_norm) {

    // Obtain the gradient of all layers
    Vector gradient;
    for (const auto& layer : network.layers) {
        auto linear_layer = std::dynamic_pointer_cast<Linear>(layer);
        if (linear_layer) {
            Vector layer_gradient = linear_layer->getGradient();
            gradient.insert(gradient.end(),
                            layer_gradient.begin(),
                            layer_gradient.end());
        }
    }

    // Compute the norm of the gradient
    double grad_norm = 0.0;
    for (const auto& elem : gradient) {
        grad_norm += elem * elem;
    }
    grad_norm = std::sqrt(grad_norm);

    // If the norm is greater than the maximum, perform the clipping
    if (grad_norm > max_norm) {
        double scale = max_norm / grad_norm;
        for (const auto& layer : network.layers) {
            auto linear_layer = std::dynamic_pointer_cast<Linear>(layer);
            if (linear_layer) {
                linear_layer->scaleGradient(scale);
            }
        }
    }
}


