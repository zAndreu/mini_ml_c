#include <stdio.h>
#include "matrix.h"

int main(void)
{
    double x1_values[] = {
        0.0,
        3.0,
        2.0,
        5.0,
        7.0,
        6.0,
        8.0,
        12.0,
        10.0,
        15.0,
        14.0,
        13.0,
        19.0,
        17.0,
        16.0,
        18.0,
        21.0,
    };

    double x2_values[] = {
        9.0,
        11.0,
        8.0,
        10.0,
        17.0,
        15.0,
        14.0,
        20.0,
        18.0,
        32.0,
        30.0,
        28.0,
        25.0,
        27.0,
        29.0,
        20.0,
        22.0,
    };
    double normalized_value = 40.0;
    size_t samples = sizeof(x1_values) / sizeof(x1_values[0]);

    matrix *x = matrix_create(samples, 2);
    matrix *y = matrix_create(samples, 1);

    for (size_t idx = 0; idx < (size_t)samples; idx++)
    {
        matrix_set(x, idx, 0, x1_values[idx] / normalized_value);
        matrix_set(x, idx, 1, x2_values[idx] / normalized_value);
        matrix_set(y, idx, 0, 2.0 * x1_values[idx] + 3.0 * x2_values[idx] + 5.0);
    }

    matrix *x_transposed = matrix_create(2, samples);
    matrix_transpose(x, x_transposed);

    matrix *weight = matrix_create(2, 1);
    matrix_fill(weight, 0.0);
    matrix *bias = matrix_create(1, 1);
    matrix_fill(bias, 0.0);

    matrix *gradient_weight = matrix_create(2, 1);
    matrix *gradient_bias = matrix_create(1, 1);

    double learning_rate = 0.01;
    size_t epochs = 10000;

    double total_loss = 0;

    matrix *prediction = matrix_create(samples, 1);
    matrix *error = matrix_create(samples, 1);
    matrix *squared_errors = matrix_create(samples, 1);

    matrix *scaled_gradient_weight = matrix_create(2, 1);
    matrix *scaled_gradient_bias = matrix_create(1, 1);

    double gradient_factor = 2.0 / (double)samples;

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        total_loss = 0;
        matrix_mult(x, weight, prediction);
        matrix_add_row_vector(prediction, bias);

        matrix_sub(prediction, y, error);
        matrix_hadamard(error, error, squared_errors);
        total_loss = matrix_mean(squared_errors);

        matrix_mult(x_transposed, error, gradient_weight);
        matrix_mult_scalar(gradient_weight, gradient_factor, gradient_weight);

        matrix *sum_errors = matrix_sum_rows(error);
        matrix_mult_scalar(sum_errors, gradient_factor, gradient_bias);
        matrix_free(sum_errors);

        matrix_mult_scalar(gradient_weight, learning_rate, scaled_gradient_weight);
        matrix_sub(weight, scaled_gradient_weight, weight);

        matrix_mult_scalar(gradient_bias, learning_rate, scaled_gradient_bias);
        matrix_sub(bias, scaled_gradient_bias, bias);

        if (epoch % 500 == 0 || epoch == epochs - 1)
        {
            printf(
                "Epoch %zu | Loss: %.10f | Weight: %.6f | Weight: %.6f | Bias: %.6f\n",
                epoch,
                total_loss,
                matrix_get(weight, 0, 0) / normalized_value,
                matrix_get(weight, 1, 0) / normalized_value,
                matrix_get(bias, 0, 0));
        }
    }

    printf("\nFinal Model Parameters:\n");
    printf("Weight 1: %.6f\n", matrix_get(weight, 0, 0) / normalized_value);
    printf("Weight 2: %.6f\n", matrix_get(weight, 1, 0) / normalized_value);
    printf("Bias: %.6f\n", matrix_get(bias, 0, 0));

    matrix *test_input = matrix_create(1, 2);
    matrix_set(test_input, 0, 0, 10.0 / normalized_value);
    matrix_set(test_input, 0, 1, 20.0 / normalized_value);

    matrix *test_prediction = matrix_create(1, 1);
    matrix_mult(test_input, weight, test_prediction);
    matrix_add_row_vector(test_prediction, bias);

    printf("\nTest Prediction for Input [10, 20]: %.6f\n", matrix_get(test_prediction, 0, 0));

    matrix_free(x);
    matrix_free(y);
    matrix_free(x_transposed);
    matrix_free(weight);
    matrix_free(bias);
    matrix_free(gradient_weight);
    matrix_free(gradient_bias);
    matrix_free(prediction);
    matrix_free(error);
    matrix_free(squared_errors);
    matrix_free(scaled_gradient_weight);
    matrix_free(scaled_gradient_bias);
    matrix_free(test_input);
    matrix_free(test_prediction);

    return 0;
}
// End of file