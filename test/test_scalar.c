#include <stdio.h>
#include "matrix.h"

#include <stdio.h>
#include <stddef.h>

int main(void)
{
    double x[] = {
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0};

    double y[] = {
        1.0,
        3.0,
        5.0,
        7.0,
        9.0,
        11.0,
        13.0,
        15.0,
        17.0,
        19.0};

    size_t samples = sizeof(x) / sizeof(x[0]);

    double weight = 0.0;
    double bias = 0.0;

    double learning_rate = 0.01;
    size_t epochs = 1000;

    double total_loss = 0;
    double gradient_weight = 0;
    double gradient_bias = 0;
    double prediction = 0;

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        total_loss = 0;
        gradient_weight = 0;
        gradient_bias = 0;
        prediction = 0;
        for (size_t idx = 0; idx < samples; idx++)
        {
            prediction = weight * x[idx] + bias;
            double error = prediction - y[idx];
            total_loss += error * error;
            gradient_weight += 2.0 * error * x[idx];
            gradient_bias += 2.0 * error;
        }
        total_loss /= samples;
        gradient_weight /= samples;
        gradient_bias /= samples;

        weight -= learning_rate * gradient_weight;
        bias -= learning_rate * gradient_bias;
    }

    printf("Peso final: %f\n", weight);
    printf("Bias final: %f\n", bias);

    double new_x = 10.0;
    double new_prediction = weight * new_x + bias;
    double new_error = new_prediction - 21.0;
    double new_total_loss = new_error * new_error;
    printf("Predicción para x = %f: %f\n", new_x, new_prediction);
    printf("Error: %f\n", new_total_loss);

    return 0;
}
// End of file