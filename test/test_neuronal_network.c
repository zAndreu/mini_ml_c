#include <stdio.h>
#include <stddef.h>

#include "matrix.h"
#include "mini_ml.h"

int main(void)
{
    double x_values[][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    double y_values[] = {
        0.0, 
        1.0, 
        1.0, 
        0.0
    };
    matrix *X = matrix_create(4, 2);
    matrix *y = matrix_create(4, 1);
    matrix *x_transposed = matrix_create(X->cols, X->rows);

    for (uint32_t i = 0; i < 4; i++)
    {
        for (uint32_t j = 0; j < 2; j++)
        {
            matrix_set(X, i, j, x_values[i][j]);
        }
        matrix_set(y, i, 0, y_values[i]);
    }

    matrix_transpose(X, x_transposed);
    matrix *W1 = matrix_create(2, 4);
    matrix *W2 = matrix_create(4, 1);
    matrix *W2_transposed = matrix_create(W2->cols, W2->rows);
    matrix *b1 = matrix_create(1, 4);
    matrix *b2 = matrix_create(1, 1);

    matrix *dW1 = matrix_create(W1->rows, W1->cols);
    matrix *dW2 = matrix_create(W2->rows, W2->cols);
    matrix *db1 = matrix_create(b1->rows, b1->cols);
    matrix *db2 = matrix_create(b2->rows, b2->cols);

    matrix_fill_random(W1, -1.0, 1.0);
    matrix_fill_random(W2, -1.0, 1.0);
    matrix_fill_random(b1, -1.0, 1.0);
    matrix_fill_random(b2, -1.0, 1.0);

    matrix *z1 = matrix_create(X->rows, W1->cols);
    matrix *z2 = matrix_create(z1->rows, W2->cols);
    matrix *dz1 = matrix_create(z1->rows, z1->cols);
    matrix *dz2 = matrix_create(z2->rows, z2->cols);
    matrix *hidden_layer = matrix_create(z1->rows, z1->cols);
    matrix *h_transposed = matrix_create(hidden_layer->cols, hidden_layer->rows);
    matrix *dhidden_layer = matrix_create(hidden_layer->rows, hidden_layer->cols);
    matrix *predictions = matrix_create(z2->rows, z2->cols);

    matrix *sigmoid_prime = matrix_create(z2->rows, z2->cols);
    matrix *relu_prime = matrix_create(z1->rows, z1->cols);

    size_t epochs = 10000;
    double learning_rate = 0.01;

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        matrix_mult(X, W1, z1);
        matrix_add_row_vector(z1, b1);

        matrix_copy(z1, hidden_layer);

        matrix_apply_function(hidden_layer, relu);

        matrix_mult(hidden_layer, W2, z2);
        matrix_add_row_vector(z2, b2);

        matrix_copy(z2, predictions);
        matrix_apply_function(predictions, sigmoid);

        // Calculamos error
        double output_loss = mean_squared_error(y, predictions);
        if (epoch % 1000 == 0 || epoch == epochs - 1)
        {
            printf("Epoch %zu, Loss: %f\n", epoch, output_loss);
        }
        matrix *output_loss_derivative = mean_squared_error_derivative(y, predictions);

        matrix_copy(z2, sigmoid_prime);
        matrix_apply_function(sigmoid_prime, sigmoid_derivative);
        matrix_hadamard(output_loss_derivative, sigmoid_prime, dz2);

        matrix_transpose(hidden_layer, h_transposed);
        matrix_mult(h_transposed, dz2, dW2);

        matrix *sum_db2 = matrix_sum_rows(dz2);
        matrix_copy(sum_db2, db2);
        matrix_free(sum_db2);

        matrix_transpose(W2, W2_transposed);
        matrix_mult(dz2, W2_transposed, dhidden_layer);

        matrix_copy(z1, relu_prime);
        matrix_apply_function(relu_prime, relu_derivative);

        matrix_hadamard(dhidden_layer, relu_prime, dz1);

        matrix_mult(x_transposed, dz1, dW1);

        matrix *sum_db1 = matrix_sum_rows(dz1);
        matrix_copy(sum_db1, db1);
        matrix_free(sum_db1);

        matrix *dw1_scaled = matrix_create(dW1->rows, dW1->cols);
        matrix_mult_scalar(dW1, learning_rate, dw1_scaled);
        matrix *dw2_scaled = matrix_create(dW2->rows, dW2->cols);
        matrix_mult_scalar(dW2, learning_rate, dw2_scaled);
        matrix *db1_scaled = matrix_create(db1->rows, db1->cols);
        matrix_mult_scalar(db1, learning_rate, db1_scaled);
        matrix *db2_scaled = matrix_create(db2->rows, db2->cols);
        matrix_mult_scalar(db2, learning_rate, db2_scaled); 

        matrix_sub(W1, dw1_scaled, W1);
        matrix_sub(W2, dw2_scaled, W2);
        matrix_sub(b1, db1_scaled, b1);
        matrix_sub(b2, db2_scaled, b2);

        matrix_free(output_loss_derivative);
        matrix_free(dw1_scaled);
        matrix_free(dw2_scaled);
        matrix_free(db1_scaled);
        matrix_free(db2_scaled);
    }


    // Comprobacion de las predicciones
    matrix_mult(X, W1, z1);
    matrix_add_row_vector(z1, b1);
    matrix_copy(z1, hidden_layer);
    matrix_apply_function(hidden_layer, relu);
    matrix_mult(hidden_layer, W2, z2);
    matrix_add_row_vector(z2, b2);
    matrix_copy(z2, predictions);
    matrix_apply_function(predictions, sigmoid);

    for (uint32_t i = 0; i < predictions->rows; i++)
    {
        double pred_value = matrix_get(predictions, i, 0);
        int pred_class = (pred_value > 0.5) ? 1 : 0;
        printf("Input: [%f, %f], Probability: %.6f, Predicted: %d, Actual: %f\n",
               matrix_get(X, i, 0), matrix_get(X, i, 1), pred_value, pred_class, matrix_get(y, i, 0));
    }

    // Liberar memoria
    matrix_free(X);
    matrix_free(y);
    matrix_free(x_transposed);
    matrix_free(W1);
    matrix_free(W2);
    matrix_free(W2_transposed);
    matrix_free(b1);
    matrix_free(b2);
    matrix_free(dW1);
    matrix_free(dW2);
    matrix_free(db1);
    matrix_free(db2);
    matrix_free(z1);
    matrix_free(z2);
    matrix_free(dz1);
    matrix_free(dz2);
    matrix_free(hidden_layer);
    matrix_free(h_transposed);
    matrix_free(dhidden_layer);
    matrix_free(predictions);
    matrix_free(sigmoid_prime);
    matrix_free(relu_prime);

    return 0;
}
// End of file