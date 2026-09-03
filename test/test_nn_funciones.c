#include <stdio.h>
#include "matrix.h"
#include "mini_ml.h"

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

    uint32_t layers[] = {2, 4, 1};
    size_t num_layers = sizeof(layers) / sizeof(layers[0]);
    activations_t activations[] = {RELU, SIGMOID};

    model_conf_t model_conf = {
        .num_layers = num_layers,
        .layers = layers,
        .activations = activations};
    model_t model = {
        .conf = &model_conf,
        .weights = NULL,
        .biases = NULL};
    training_conf_t conf = {
        .model = &model,
        .X = X,
        .y = y,
        .loss_function = MEAN_SQUARED_ERROR,
        .optimizer = SGD,
        .learning_rate = 0.1,
        .epochs = 6000};

    training_config(&conf);
    model_t *trained_model = model_fit();
    if (trained_model == NULL)
    {
        fprintf(stderr, "Error: Model training failed.\n");
        matrix_free(X);
        matrix_free(y);
        return 1;
    }

    matrix *predictions = model_predict(trained_model, X);
    printf("Predictions:\n");
    for (uint32_t i = 0; i < predictions->rows; i++)
    {
        printf("Input: [%f, %f] - Predicted: %f\n",
               matrix_get(X, i, 0), matrix_get(X, i, 1),
               matrix_get(predictions, i, 0));
    }
    model_free(trained_model);
    matrix_free(predictions);

    matrix_free(X);
    matrix_free(y);
    return 0;
}
// End of file