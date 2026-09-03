#ifndef MINI_ML_H
#define MINI_ML_H

#include "matrix.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef enum
{
    RELU,
    SIGMOID
} activations_t;

typedef enum
{
    MEAN_SQUARED_ERROR,
    BINARY_CROSS_ENTROPY
} loss_t;

typedef enum
{
    SGD,
    ADAM
} optimizer_t;

typedef struct
{
    size_t num_layers;
    uint32_t *layers;
    activations_t *activations;
} model_conf_t;

typedef struct
{
    model_conf_t *conf;
    matrix **weights;
    matrix **biases;
} model_t;

typedef struct
{
    model_t *model;
    matrix *X;
    matrix *y;
    loss_t loss_function;
    optimizer_t optimizer;
    double learning_rate;
    uint32_t epochs;
} training_conf_t;

// Funciones de activacion
double relu(double x);
double relu_derivative(double x);

double sigmoid(double x);
double sigmoid_derivative(double x);

// Funciones de perdida
double mean_squared_error(matrix *y_true, matrix *y_pred);
matrix *mean_squared_error_derivative(matrix *y_true, matrix *y_pred);

double binary_cross_entropy(matrix *y_true, matrix *y_pred);
matrix *binary_cross_entropy_derivative(matrix *y_true, matrix *y_pred);

void training_config(training_conf_t *config);

model_t *model_fit(void);
matrix *model_predict(model_t *model, matrix *X);

void model_free(model_t *model);
#endif // MINI_ML_H