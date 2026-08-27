// Fallos: No logra generalizar bien por falta de datos, para barreras muy pequeñas o muy altas no logra un correcto funcionamiento.

#include <stdio.h>
#include "matrix.h"
#include "mini_ml.h"

int main(void)
{
    const size_t samples = 255;
    const double max_value = 30.0;
    // Funcion a buscar: 2w1 + 3w2 + 5
    double barrier_num = 83.0;
    matrix *x = matrix_create(samples, 2);
    matrix *y = matrix_create(samples, 1);

    matrix_fill_random(x, 0.0, max_value / max_value);
    for (size_t idx = 0; idx < samples; idx++)
    {
        const double output = (((2.0 * matrix_get(x, idx, 0) * max_value) + (3.0 * matrix_get(x, idx, 1) * max_value)) + 5.0) >= barrier_num ? 1.0 : 0.0;
        matrix_set(y, idx, 0, (output));
    }
    matrix *x_transposed = matrix_create(2, samples);
    matrix_transpose(x, x_transposed);

    matrix *weight = matrix_create(2, 1);
    matrix_fill(weight, 0.0);

    matrix *bias = matrix_create(1, 1);
    matrix_fill(bias, 0.0);

    matrix *prediction = matrix_create(samples, 1);
    matrix *error = matrix_create(samples, 1);
    matrix *gradient_weight = matrix_create(2, 1);
    matrix *gradient_bias = matrix_create(1, 1);

    const size_t epochs = 60000;
    double loss = 0.0;
    const double gradient_value = 1.0 / samples;
    const double learaning_rate = 0.2;

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        matrix_mult(x, weight, prediction);
        matrix_add_row_vector(prediction, bias);

        matrix_apply_function(prediction, sigmoid);
        loss = binary_cross_entropy(y, prediction);

        matrix_sub(prediction, y, error);
        matrix *auxiliar_weight = matrix_create(x_transposed->rows, error->cols);
        matrix_mult(x_transposed, error, auxiliar_weight);
        matrix_mult_scalar(auxiliar_weight, gradient_value, gradient_weight);
        matrix_free(auxiliar_weight);

        matrix *auxiliar_bias = matrix_sum_rows(error);
        matrix_mult_scalar(auxiliar_bias, gradient_value, gradient_bias);
        matrix_free(auxiliar_bias);

        // Actualizar
        matrix *temp_weight = matrix_create(weight->rows, weight->cols);
        matrix_copy(weight, temp_weight);
        matrix_mult_scalar(gradient_weight, learaning_rate, gradient_weight);
        matrix_sub(temp_weight, gradient_weight, weight);
        matrix_free(temp_weight);

        matrix *temp_bias = matrix_create(bias->rows, bias->cols);
        matrix_copy(bias, temp_bias);
        matrix_mult_scalar(gradient_bias, learaning_rate, gradient_bias);
        matrix_sub(temp_bias, gradient_bias, bias);
        matrix_free(temp_bias);

        if (epoch % 1000 == 0)
        {
            fprintf(stdout, "%f\n", loss);
        }
    }
    const double valor_x1 = 21.0;
    const double valor_x2 = 12.0;
    matrix *test_input = matrix_create(1, 2);
    matrix_set(test_input, 0, 0, valor_x1 / max_value);
    matrix_set(test_input, 0, 1, valor_x2 / max_value);
    matrix *test_output = matrix_create(1, 1);

    matrix_mult(test_input, weight, test_output);
    matrix_add_row_vector(test_output, bias);

    matrix_apply_function(test_output, sigmoid);
    const double valor = 2.0 * valor_x1 + 3.0 * valor_x2 + 5.0;
    fprintf(stdout, "Valor esperado: %f\n", valor);
    fprintf(stdout, "Prediccion: %f\n", matrix_get(test_output, 0, 0));

    matrix_free(x);
    matrix_free(y);
    matrix_free(x_transposed);
    matrix_free(weight);
    matrix_free(bias);
    matrix_free(prediction);
    matrix_free(error);
    matrix_free(gradient_weight);
    matrix_free(gradient_bias);

    matrix_free(test_input);
    matrix_free(test_output);

    return 0;
}
// End of file