#include <stdio.h>
#include "matrix.h"
#include "mini_ml.h"

extern training_conf_t *train_conf;

int main(void)
{
    double x_values[][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}};
    double y_values[] = {
        0.0,
        1.0,
        1.0,
        0.0};
    matrix *X = matrix_create(4, 2);
    matrix *y = matrix_create(4, 1);

    for (uint32_t i = 0; i < 4; i++)
    {
        for (uint32_t j = 0; j < 2; j++)
        {
            matrix_set(X, i, j, x_values[i][j]);
        }
        matrix_set(y, i, 0, y_values[i]);
    }
    model_conf_t model_conf = {
        .num_layers = 3,
        .layers = (uint32_t[]){2, 4, 1},
        .activations = (activations_t[]){RELU, SIGMOID}};
    model_t model = {
        .conf = &model_conf,
        .weights = (matrix *[]){matrix_create(2, 4), matrix_create(4, 1)},
        .biases = (matrix *[]){matrix_create(1, 4), matrix_create(1, 1)}};

    matrix_fill_random(model.weights[0], -1.0, 1.0);
    matrix_fill_random(model.weights[1], -1.0, 1.0);
    matrix_fill(model.biases[0], 0.0);
    matrix_fill(model.biases[1], 0.0);

    training_conf_t conf = {
        .model = &model,
        .X = X,
        .y = y,
        .loss_function = MEAN_SQUARED_ERROR,
        .optimizer = SGD,
        .learning_rate = 0.1,
        .epochs = 6000};
    train_conf = &conf;
    model_fit();

    matrix *predictions = model_predict(X);
    printf("Predictions:\n");
    for (uint32_t i = 0; i < predictions->rows; i++)
    {
        printf("Input: [%f, %f] - Predicted: %f\n",
               matrix_get(X, i, 0), matrix_get(X, i, 1),
               matrix_get(predictions, i, 0));
    }

    model_clear_cache();
    matrix_free(model.weights[0]);
    matrix_free(model.weights[1]);
    matrix_free(model.biases[0]);
    matrix_free(model.biases[1]);
    matrix_free(X);
    matrix_free(y);
    return 0;
}
// End of file