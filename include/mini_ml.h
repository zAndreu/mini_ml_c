#ifndef MINI_ML_H
#define MINI_ML_H

#include "matrix.h"

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

#endif // MINI_ML_H