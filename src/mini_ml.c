#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stddef.h>

#include "matrix.h"
#include "mini_ml.h"

training_conf_t *train_conf = NULL;
static matrix **out = NULL;
static matrix **pre_activation = NULL;
static matrix **dWs = NULL;
static matrix **dbs = NULL;
static size_t buffer_layers = 0;

static void free_buffers(void)
{
    for (size_t layer_idx = 0; layer_idx < buffer_layers; layer_idx++)
    {
        matrix_free(out[layer_idx]);
        matrix_free(pre_activation[layer_idx]);
        matrix_free(dWs[layer_idx]);
        matrix_free(dbs[layer_idx]);
    }
    free(out);
    free(pre_activation);
    free(dWs);
    free(dbs);
    out = NULL;
    pre_activation = NULL;
    dWs = NULL;
    dbs = NULL;
    buffer_layers = 0;
}

static bool allocate_buffers(size_t layer_count)
{
    if (buffer_layers == layer_count && out != NULL)
    {
        return true;
    }

    free_buffers();
    out = calloc(layer_count, sizeof(*out));
    pre_activation = calloc(layer_count, sizeof(*pre_activation));
    dWs = calloc(layer_count, sizeof(*dWs));
    dbs = calloc(layer_count, sizeof(*dbs));
    if (out == NULL || pre_activation == NULL || dWs == NULL || dbs == NULL)
    {
        free_buffers();
        return false;
    }
    buffer_layers = layer_count;
    return true;
}

// Funciones de activacion
double relu(double x)
{
    return (x > 0.0) ? x : 0;
}

double relu_derivative(double x)
{
    return (x > 0.0) ? 1 : 0;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x)
{
    double aux = sigmoid(x);
    return (aux * (1 - aux));
}

// Funciones de perdida

double mean_squared_error(matrix *y_true, matrix *y_pred)
{
    double loss = 0.0;
    uint32_t size = matrix_size(y_true);
    for (size_t idx = 0; idx < y_true->rows; idx++)
    {
        for (size_t idy = 0; idy < y_true->cols; idy++)
        {
            double error = matrix_get(y_pred, idx, idy) - matrix_get(y_true, idx, idy);
            loss += error * error;
        }
    }
    return loss / (double)size;
}

matrix *mean_squared_error_derivative(matrix *y_true, matrix *y_pred)
{
    matrix *derivative = matrix_create(y_true->rows, y_true->cols);
    const double scale = 2.0 / matrix_size(y_true);
    for (size_t idx = 0; idx < y_true->rows; idx++)
    {
        for (size_t idy = 0; idy < y_true->cols; idy++)
        {
            double value = scale * (matrix_get(y_pred, idx, idy) - matrix_get(y_true, idx, idy));
            matrix_set(derivative, idx, idy, value);
        }
    }
    return derivative;
}

double binary_cross_entropy(matrix *y_true, matrix *y_pred)
{
    const double epsilon = 1e-15;

    double loss = 0.0;
    for (size_t i = 0; i < y_true->rows; i++)
    {
        for (size_t j = 0; j < y_true->cols; j++)
        {
            double y_t = matrix_get(y_true, i, j);
            double y_p = matrix_get(y_pred, i, j);
            if (y_p < epsilon)
            {
                y_p = epsilon;
            }
            if (y_p > 1.0 - epsilon)
            {
                y_p = 1.0 - epsilon;
            }

            loss += y_t * log(y_p) + (1.0 - y_t) * log(1 - y_p);
        }
    }

    return -loss / (double)matrix_size(y_true);
}

matrix *binary_cross_entropy_derivative(matrix *y_true, matrix *y_pred)
{
    const double epsilon = 1e-15;
    matrix *derivative = matrix_create(y_true->rows, y_true->cols);
    const double scale = 1.0 / (double)matrix_size(y_true);
    for (size_t idx = 0; idx < y_true->rows; idx++)
    {
        for (size_t idy = 0; idy < y_true->cols; idy++)
        {
            double y_t = matrix_get(y_true, idx, idy);
            double y_p = matrix_get(y_pred, idx, idy);
            if (y_p < epsilon)
            {
                y_p = epsilon;
            }
            if (y_p > 1.0 - epsilon)
            {
                y_p = 1.0 - epsilon;
            }
            double value = scale * ((y_p - y_t) / ((y_p) * (1.0 - y_p)));
            matrix_set(derivative, idx, idy, value);
        }
    }
    return derivative;
}

matrix *model_predict(matrix *X)
{
    if (train_conf == NULL || train_conf->model == NULL || X == NULL)
    {
        fprintf(stderr, "Error: No model is available for prediction.\n");
        return NULL;
    }

    model_t *model = train_conf->model;
    if (model->conf == NULL || model->conf->num_layers < 2)
    {
        fprintf(stderr, "Error: The model must have at least two layers.\n");
        return NULL;
    }

    size_t layer_count = model->conf->num_layers - 1;
    if (!allocate_buffers(layer_count))
    {
        fprintf(stderr, "Error: Could not allocate model buffers.\n");
        return NULL;
    }
    for (size_t layer_idx = 0; layer_idx < layer_count; layer_idx++)
    {
        matrix_free(out[layer_idx]);
        matrix_free(pre_activation[layer_idx]);
        out[layer_idx] = NULL;
        pre_activation[layer_idx] = NULL;
    }

    matrix *current_input = X;

    for (size_t layer_idx = 0; layer_idx < layer_count; layer_idx++)
    {
        matrix *z = matrix_create(current_input->rows, model->weights[layer_idx]->cols);
        if (z == NULL || !matrix_mult(current_input, model->weights[layer_idx], z) ||
            !matrix_add_row_vector(z, model->biases[layer_idx]))
        {
            matrix_free(z);
            return NULL;
        }

        pre_activation[layer_idx] = z;
        matrix *activated = matrix_create(z->rows, z->cols);
        if (activated == NULL || !matrix_copy(z, activated))
        {
            matrix_free(activated);
            return NULL;
        }

        // Apply activation function
        if (model->conf->activations[layer_idx] == RELU)
        {
            matrix_apply_function(activated, relu);
        }
        else if (model->conf->activations[layer_idx] == SIGMOID)
        {
            matrix_apply_function(activated, sigmoid);
        }
        else
        {
            fprintf(stderr, "Error: Unknown activation function.\n");
        }

        current_input = activated;
        out[layer_idx] = activated;
    }

    return current_input; // Return the final output
}

double model_loss(matrix *y_true, matrix *y_pred, matrix *loss_derivative)
{
    if (train_conf == NULL)
    {
        fprintf(stderr, "Error: No training configuration available.\n");
        return -1.0;
    }

    double loss = 0.0;
    if (train_conf->loss_function == MEAN_SQUARED_ERROR)
    {
        loss = mean_squared_error(y_true, y_pred);
        matrix *derivative = mean_squared_error_derivative(y_true, y_pred);
        matrix_copy(derivative, loss_derivative);
        matrix_free(derivative);
    }
    else if (train_conf->loss_function == BINARY_CROSS_ENTROPY)
    {
        loss = binary_cross_entropy(y_true, y_pred);
        matrix *derivative = binary_cross_entropy_derivative(y_true, y_pred);
        matrix_copy(derivative, loss_derivative);
        matrix_free(derivative);
    }
    else
    {
        fprintf(stderr, "Error: Unknown loss function.\n");
        return -1.0;
    }

    return loss;
}

void model_backward(matrix *loss_derivative)
{
    if (train_conf == NULL || train_conf->model == NULL)
    {
        fprintf(stderr, "Error: No model is available for backpropagation.\n");
        return;
    }

    if (loss_derivative == NULL || train_conf->model->conf == NULL ||
        train_conf->model->conf->num_layers < 2)
    {
        fprintf(stderr, "Error: Invalid model or loss derivative.\n");
        return;
    }

    size_t layer_count = train_conf->model->conf->num_layers - 1;
    bool owns_loss_derivative = false;
    for (size_t layer_idx = layer_count; layer_idx-- > 0;)
    {
        matrix *current_weight = train_conf->model->weights[layer_idx];
        matrix *z = pre_activation[layer_idx];
        matrix *dZ = matrix_create(loss_derivative->rows, loss_derivative->cols);
        if (dZ == NULL)
        {
            return;
        }

        for (uint32_t row = 0; row < z->rows; row++)
        {
            for (uint32_t col = 0; col < z->cols; col++)
            {
                double derivative = train_conf->model->conf->activations[layer_idx] == RELU
                                        ? relu_derivative(matrix_get(z, row, col))
                                        : sigmoid_derivative(matrix_get(z, row, col));
                matrix_set(dZ, row, col, matrix_get(loss_derivative, row, col) * derivative);
            }
        }

        matrix *layer_input = layer_idx == 0 ? train_conf->X : out[layer_idx - 1];
        matrix *input_transposed = matrix_create(layer_input->cols, layer_input->rows);
        if (input_transposed == NULL || !matrix_transpose(layer_input, input_transposed))
        {
            matrix_free(input_transposed);
            matrix_free(dZ);
            return;
        }

        // Compute gradients for weights and biases
        matrix_free(dWs[layer_idx]);
        dWs[layer_idx] = matrix_create(input_transposed->rows, dZ->cols);
        matrix_mult(input_transposed, dZ, dWs[layer_idx]);

        matrix_free(dbs[layer_idx]);
        dbs[layer_idx] = matrix_create(1, dZ->cols);
        for (size_t j = 0; j < dZ->cols; j++)
        {
            double sum = 0.0;
            for (size_t i = 0; i < dZ->rows; i++)
            {
                sum += matrix_get(dZ, i, j);
            }
            matrix_set(dbs[layer_idx], 0, j, sum);
        }
        matrix_free(input_transposed);

        // Compute the derivative for the next layer
        if (layer_idx > 0)
        {
            matrix *weight_transposed = matrix_create(current_weight->cols, current_weight->rows);
            matrix *next_loss_derivative = matrix_create(loss_derivative->rows, current_weight->rows);
            if (weight_transposed == NULL || next_loss_derivative == NULL ||
                !matrix_transpose(current_weight, weight_transposed) ||
                !matrix_mult(dZ, weight_transposed, next_loss_derivative))
            {
                matrix_free(weight_transposed);
                matrix_free(next_loss_derivative);
                matrix_free(dZ);
                return;
            }
            matrix_free(weight_transposed);
            if (owns_loss_derivative)
            {
                matrix_free(loss_derivative);
            }
            matrix_free(dZ);
            loss_derivative = next_loss_derivative;
            owns_loss_derivative = true;
        }
        else
        {
            matrix_free(dZ);
        }
    }
    if (owns_loss_derivative)
    {
        matrix_free(loss_derivative);
    }
}

void model_clear_cache(void)
{
    free_buffers();
}

void model_optimize(void)
{
    if (train_conf == NULL || train_conf->model == NULL)
    {
        fprintf(stderr, "Error: No model is available for optimization.\n");
        return;
    }

    if (train_conf->learning_rate <= 0.0)
    {
        fprintf(stderr, "Error: Learning rate must be positive.\n");
        return;
    }
    size_t num_layers = train_conf->model->conf->num_layers;
    if (train_conf->optimizer == SGD)
    {
        for (uint32_t layer_idx = 0; layer_idx < (num_layers - 1); layer_idx++)
        {
            matrix *weight = train_conf->model->weights[layer_idx];
            matrix *bias = train_conf->model->biases[layer_idx];
            matrix *dW = dWs[layer_idx];
            matrix *db = dbs[layer_idx];

            matrix *scaled_dW = matrix_create(dW->rows, dW->cols);
            matrix_mult_scalar(dW, train_conf->learning_rate, scaled_dW);
            matrix *scaled_db = matrix_create(db->rows, db->cols);
            matrix_mult_scalar(db, train_conf->learning_rate, scaled_db);
            matrix_free(dW);
            matrix_free(db);
            dWs[layer_idx] = NULL;
            dbs[layer_idx] = NULL;

            matrix_sub(weight, scaled_dW, weight);
            matrix_sub(bias, scaled_db, bias);

            matrix_free(scaled_dW);
            matrix_free(scaled_db);
        }
    }
    else if (train_conf->optimizer == ADAM)
    {
        /* TODO: Implementar el optimizador ADAM 
        * Se necesitan las dos betas y el epsilon: b1 = 0.9, b2 = 0.999, epsilon = 1e-8
        * Dos buffers m y v
        * Y un contador de iteraciones t
        */
        fprintf(stderr, "Error: ADAM optimizer is not implemented yet.\n");
    }
    else
    {
        fprintf(stderr, "Error: Unknown optimizer.\n");
    }
}

void model_fit(void)
{
    if (train_conf == NULL || train_conf->model == NULL)
    {
        fprintf(stderr, "Error: No model is available for training.\n");
        return;
    }

    for (uint32_t epoch = 0; epoch < train_conf->epochs; epoch++)
    {
        matrix *predictions = model_predict(train_conf->X);
        matrix *loss_derivative = matrix_create(predictions->rows, predictions->cols);
        double loss = model_loss(train_conf->y, predictions, loss_derivative);
        if (epoch % (train_conf->epochs / 10) == 0 || epoch == train_conf->epochs - 1)
        {
            printf("Epoch %u/%u - Loss: %.6f\n", epoch + 1, train_conf->epochs, loss);
        }

        model_backward(loss_derivative);
        model_optimize();

        matrix_free(loss_derivative);
    }
}
// End of file