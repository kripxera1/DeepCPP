/* 
 * File: examples/classifier.cpp
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Contains the code for the MNIST Classifier example project.
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
                      string name , string path);

int main(){
    int seed= 123;
    srand(seed);
    omp_set_num_threads(8);

    string train_data_path = "data/mnist_train.txt";
    string test_dataPath = "data/mnist_test.txt";
    string images_path = "images/classifier/";

    //Hyperparameter initialization
    /////////////////////////////////////////////////////////////////////////

    vector<shared_ptr<Layer>> layers;
    shared_ptr<LossFunction> loss;
    shared_ptr<Optimizer> optimizer;

    layers = {
        make_shared<Linear>(784, 128),
        make_shared<LeakyRelu>(0.05),
        make_shared<Linear>(128, 64),
        make_shared<LeakyRelu>(0.05),
        make_shared<Linear>(64, 32),
        make_shared<LeakyRelu>(0.05),
        make_shared<Linear>(32, 10),
        make_shared<SoftMax>()
    };

    loss = make_shared<CrossEntropy>();

    optimizer =  make_shared<Adam>();

    double learn_rate = 0.001;
    ConstantLearningRate lr_schedule(learn_rate);


    NeuralNetwork nn(layers, loss, optimizer);


    int batch_size = 20;
    int num_epochs = 20;

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


    // Load test dataset

    cout << "\n\nLoading test dataset... " << flush;

    auto test_data = loadData(test_dataPath.c_str());
    int testSize = test_data.size();
    int nBatchTest = testSize/batch_size;

    cout << "\tDone!"
         << "\n\nNumber of test examples: "
         << testSize << endl;


    // Training
    /////////////////////////////////////////////////////////////////////////

    Matrix X;    
    Matrix Y;
    Matrix Y_hat;
    cout << "\n\nTraining:\n" << endl;
    for(int epoch = 0; epoch < num_epochs; epoch++){

        double training_accuracy = 0;
        double test_accuracy = 0;
        double training_loss = 0;
        double test_loss = 0;

        //Training
        for(int it = 0; it < num_batch_train; it++){
            // Load batch
            loadBatch(train_data,batch_size,it,X,Y);
            // Pass forward
            Y_hat = nn.forward(X);
            // Accuracy and loss update
            training_accuracy += getAccuracy(Y_hat,Y)/num_batch_train;
            training_loss += nn.loss->compute(Y_hat,Y)/num_batch_train;
            // Pass backward
            nn.backward(Y_hat,Y);
            // Gradient clipping
            gradientClipping(nn,5);
            // Update weights
            learn_rate = lr_schedule.getLearningRate(epoch);
            nn.update(learn_rate, batch_size);
        }

        //Testing
        for(int it = 0; it < nBatchTest; it++){
            // Load batch
            loadBatch(test_data,batch_size,it,X,Y);
            // Pass forward
            Y_hat = nn.forward(X);
            // Accuracy and loss update
            test_accuracy += getAccuracy(Y_hat,Y)/nBatchTest;
            test_loss += nn.loss->compute(Y_hat,Y)/nBatchTest;
        }

        cout << setprecision(4) << fixed  
             << "\tEpoch " << to_string(epoch) 
             << ":\tTrain acc: " << training_accuracy
             << "\tTest acc: " << test_accuracy
             << "\tTrain loss: " << training_loss 
             << "\tTest loss: " << test_loss << endl;
    }
    cout << "\nTraining done!\n" << endl;
    cout << "Saving image classification examples" << endl;


    //Image classification example
    /////////////////////////////////////////////////////////////////////////

    // number of images to classify
    int nImages = 20;
    int height = 28;
    int width  = 28;
    int index = 0;

    loadBatch(test_data,nImages,index,X,Y);

    //Pass forward

    Y_hat = nn.forward(X);
    auto prediction = getPrediction(Y_hat);
    
    for(int images = 0; images < nImages; images++){
        saveImageSamples(X,height,width,images,images+1,
                         to_string(prediction[images]),images_path);
    }
    
    return 0;
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