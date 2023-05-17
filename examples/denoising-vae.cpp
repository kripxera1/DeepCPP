/* 
 * File: examples/denoising-vae.cpp
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Contains the code for the MNIST Denoising VAE example project.
 */

#include <iostream>
#include <iomanip>
#include <memory>
#include <omp.h>
#include <random>

#include "bitmap.h"
#include "NNUtils.h"
#include "typedefs.h"

using namespace std;

void saveImageSamples(const Matrix& samples,
                      int height, int width, int begin, int end,
                      string name , string path );

Matrix addNormalNoise(const Matrix& X, double mean, double std_dev, int it, int epoch);

int main(){
    int seed= 123;
    srand(seed);
    omp_set_num_threads(8);

    string train_data_path = "data/mnist_train.txt";
    string images_path = "images/denoising-vae/";

    //Hyperparameter initialization
    /////////////////////////////////////////////////////////////////////////

    vector<shared_ptr<Layer>> layers;
    shared_ptr<LossFunction> loss;
    shared_ptr<Optimizer> optimizer;

    layers = {
        make_shared<Linear>(784, 128),
        make_shared<Sigmoid>(),
        make_shared<Linear>(128, 64),
        make_shared<NormalSampling>(), // Latent space of 32 dimensions
        make_shared<Linear>(32, 128),
        make_shared<Sigmoid>(),
        make_shared<Linear>(128, 784),
        make_shared<Sigmoid>(),
    };

    loss = make_shared<MeanSquaredError>();

    optimizer =  make_shared<Adam>();

    double learn_rate = 0.002;
    ConstantLearningRate lr_schedule(learn_rate);

    NeuralNetwork nn(layers, loss, optimizer);
    
    int batch_size = 20;
    int num_epochs = 200;

    cout << "Hyperparameters:\n"
         << "\n\tLearning rate:\t\t" << learn_rate
         << "\n\tBatch size:\t\t" << batch_size
         << "\n\tNumber of epochs:\t" << num_epochs
         << "\n" << endl;


    // Data loading
    /////////////////////////////////////////////////////////////////////////

    // Load training dataset

    cout << "\nLoading train dataset..." << flush;

    vector<vector<int>> train_data=loadData(train_data_path.c_str());
    int train_size = train_data.size();
    int num_batch_train = train_size/batch_size;
    
    cout << "\tDone!"
         << "\n\nNumber of train examples: "
         << train_size << endl;

    int height = 28;
    int width  = 28;

    // Training
    /////////////////////////////////////////////////////////////////////////
      
    Matrix Y;
    Matrix Y_hat;
    Matrix X;  
    Matrix X_noise;
    cout << "\n\nTraining:\n" << endl;
    for(int epoch = 0; epoch < num_epochs; epoch++){

        double training_loss = 0;

        //Training
        for(int it = 0; it < num_batch_train; it++){
            // Load batch
            loadBatch(train_data,batch_size,it,X,Y);
            // Add noise to the input data
            X_noise = addNormalNoise(X, 0.1307 , 0.3081, it, epoch);
            // Pass forward
            Y_hat = nn.forward(X_noise);
            // Loss update
            training_loss += nn.loss->compute(Y_hat,X)/num_batch_train;
            // Pass backward
            nn.backward(Y_hat,X);
            // Gradient clipping
            gradientClipping(nn,5);
            // Update weights
            learn_rate = lr_schedule.getLearningRate(epoch);
            nn.update(learn_rate, batch_size);

            // Saving image samples
            if(it%50 == 0){
                saveImageSamples(Y_hat, height, width, 0, batch_size,
                                 "denoised", images_path);
                saveImageSamples(X_noise, height, width, 0, batch_size,
                                 "noisy", images_path);
            }
        }

        cout << setprecision(4) << fixed  
             << "\tEpoch " << to_string(epoch) 
             << "\tTrain loss: " << training_loss << endl ;
    }
    cout << "\nTraining done!\n" << endl;

    return 0;
}


Matrix addNormalNoise(const Matrix& X, double mean, double std_dev, int it, int epoch) {
    Matrix X_noise = X;
    std::default_random_engine generator(1234);
    std::normal_distribution<double> distribution(mean, std_dev);
    for (int i = 0; i < X_noise.size(); i++) {
        generator.seed(generator() + it + epoch);
        for (int j = 0; j < X_noise[0].size(); j++) {
            X_noise[i][j] += distribution(generator);
            X_noise[i][j] = std::min(X_noise[i][j], 1.0);
        }
    }
    return X_noise;
}


void saveImageSamples(const Matrix& samples,
                      int height, int width, int begin, int end,
                      string name , string path) {

    auto image = reserveSpaceImage(height, width);

    for (int images = begin; images < end; images++) {
        auto it_recon = T(samples)[images].begin();
        for (int i = height - 1; i >= 0; i--) {
            for (int j = 0; j < width; j++) {
                unsigned char value = (unsigned char)(int)((*it_recon) * 255);
                image[i][j][0] = value;
                image[i][j][1] = value;
                image[i][j][2] = value;
                it_recon++;
            }
        }

        string image_name =  path + "/" +to_string(images) + "_" + name + ".bmp";
        generateBitmapImage(image, height, width, (char*)image_name.c_str());
    }

    freeSpaceImage(image, height, width);
}