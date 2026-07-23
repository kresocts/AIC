#include <cmath>
#include <iostream>

using namespace std;

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double output)
{
    return output * (1.0 - output);
}

int main()
{

    // INPUT
    double inputs[3] = { 1.0, 0.5, 0.2 };


    // PRVI NEURON
    double weights1[3] = { 0.8, 0.4, 0.3 };
    double bias1 = 0.1;

    // DRUGI NEURON
    double weights2[3] = { -0.2, 0.9, -0.5 };
    double bias2 = 0.3;


    // IZLAZNI NEURON
    double outputWeights[2] = { 0.8, -0.3 };
    double outputBias = 0.2;

    double target = 0.0;
    double learningRate = 0.1;

    for (int epoch = 0; epoch < 100; ++epoch)
    {

        double sum1 = bias1;
        double sum2 = bias2;

        for (int i = 0; i < 3; ++i)
        {
            sum1 += inputs[i] * weights1[i];
            sum2 += inputs[i] * weights2[i];
        }

        double hiddenOutputs[2];

        hiddenOutputs[0] = sigmoid(sum1);
        hiddenOutputs[1] = sigmoid(sum2);


        // Forward pass izlaznog neurona
        double outputSum = outputBias;

        for (int i = 0; i < 2; ++i)
        {
            outputSum += hiddenOutputs[i] * outputWeights[i];
        }

        double finalOutput = sigmoid(outputSum);

        // Greška
        double error = finalOutput - target;
        double loss = 0.5 * error * error;

        // Smjer promjene
        double outputDelta =
            error * sigmoidDerivative(finalOutput);


        double hiddenDeltas[2];

        for (int i = 0; i < 2; ++i)
        {
            hiddenDeltas[i] =
                outputDelta *
                outputWeights[i] *
                sigmoidDerivative(hiddenOutputs[i]);
        }

        for (int i = 0; i < 3; ++i)
        {
            weights1[i] -=
                learningRate * hiddenDeltas[0] * inputs[i];

            weights2[i] -=
                learningRate * hiddenDeltas[1] * inputs[i];
        }

        bias1 -= learningRate * hiddenDeltas[0];
        bias2 -= learningRate * hiddenDeltas[1];

        // Promjena težina
        for (int i = 0; i < 2; ++i)
        {
            double weightGradient =
                outputDelta * hiddenOutputs[i];

            outputWeights[i] -=
                learningRate * weightGradient;
        }

        // Promjena biasa
        outputBias -= learningRate * outputDelta;

        if (epoch == 0)
        {
            cout << "Hidden delta 0: " << hiddenDeltas[0] << '\n';
            cout << "Hidden delta 1: " << hiddenDeltas[1] << '\n';
            cout << "New hidden weight 1[0]: " << weights1[0] << '\n';
            cout << "New hidden weight 2[0]: " << weights2[0] << '\n';
            cout << "New hidden bias 1: " << bias1 << '\n';
            cout << "New hidden bias 2: " << bias2 << '\n';
        }

        if (epoch % 10 == 0 || epoch == 99)
        {
            cout << "Epoch " << epoch
                << " | Output: " << finalOutput
                << " | Loss: " << loss << '\n';
        }
    }
}