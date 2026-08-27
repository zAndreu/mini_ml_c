#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stddef.h>

#include "matrix.h"
#include "mini_ml.h"

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
// End of file