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
    double trainingInputs[2][3] =
    {
        { 1.0, 0.5, 0.2 },
        { 0.0, 0.2, 0.9 }
    };

    double trainingTargets[2] =
    {
        1.0,
        0.0
    };

    // PRVI NEURON
    double weights1[3] = { 0.8, 0.4, 0.3 };
    double bias1 = 0.1;

    // DRUGI NEURON
    double weights2[3] = { -0.2, 0.9, -0.5 };
    double bias2 = 0.3;
   

    // IZLAZNI NEURON
    double outputWeights[2] = { 0.8, -0.3 };
    double outputBias = 0.2;

    double learningRate = 0.1;

    for (int epoch = 0; epoch < 5000; ++epoch)
    {
        double totalLoss = 0.0;

        for (int sample = 0; sample < 2; ++sample)
        {
            double* inputs = trainingInputs[sample];
            double target = trainingTargets[sample];

            // Skriveni neuroni
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

            // Izlazni neuron
            double outputSum = outputBias;

            for (int i = 0; i < 2; ++i)
            {
                outputSum += hiddenOutputs[i] * outputWeights[i];
            }

            double finalOutput = sigmoid(outputSum);

            // Greška
            double error = finalOutput - target;
            double loss = 0.5 * error * error;

            totalLoss += loss;

            // Backpropagation
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

            // Promjena skrivenih težina
            for (int i = 0; i < 3; ++i)
            {
                weights1[i] -=
                    learningRate * hiddenDeltas[0] * inputs[i];

                weights2[i] -=
                    learningRate * hiddenDeltas[1] * inputs[i];
            }

            bias1 -= learningRate * hiddenDeltas[0];
            bias2 -= learningRate * hiddenDeltas[1];

            // Promjena izlaznih težina
            for (int i = 0; i < 2; ++i)
            {
                outputWeights[i] -=
                    learningRate * outputDelta * hiddenOutputs[i];
            }

            outputBias -= learningRate * outputDelta;

            if (epoch % 500 == 0 || epoch == 4999) {
                cout << "Epoch " << epoch
                    << " | Sample " << sample
                    << " | Target: " << target
                    << " | Output: " << finalOutput << '\n';
            }
        }

        if (epoch % 500 == 0 || epoch == 4999) {
            cout << "Average loss: " << totalLoss / 2.0 << "\n\n";
        }
    }
}