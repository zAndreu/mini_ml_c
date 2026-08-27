#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"

matrix *matrix_create(uint32_t rows, uint32_t cols)
{
    if (rows == 0 || cols == 0)
    {
        return NULL;
    }
    matrix *m = (matrix *)malloc(sizeof(matrix));
    if (m == NULL)
    {
        return NULL;
    }
    m->rows = rows;
    m->cols = cols;
    m->data = (double *)malloc(rows * cols * sizeof(double));
    if (m->data == NULL)
    {
        free(m);
        return NULL;
    }
    return m;
}

void matrix_free(matrix *m)
{
    if (m == NULL)
    {
        return;
    }
    free(m->data);
    free(m);
}

double matrix_get(const matrix *m, uint32_t row, uint32_t col)
{
    return m->data[row * m->cols + col];
}

void matrix_set(matrix *m, uint32_t row, uint32_t col, double value)
{
    m->data[row * m->cols + col] = value;
}

void matrix_fill(matrix *m, double value)
{
    for (uint32_t x_row = 0; x_row < m->rows; x_row++)
    {
        for (uint32_t y_col = 0; y_col < m->cols; y_col++)
        {
            matrix_set(m, x_row, y_col, value);
        }
    }
}

void matrix_fill_random(matrix *m, double min, double max)
{
    for (uint32_t x_row = 0; x_row < m->rows; x_row++)
    {
        for (uint32_t y_col = 0; y_col < m->cols; y_col++)
        {
            double random_value = ((double)rand() / RAND_MAX) * (max - min) + min;
            matrix_set(m, x_row, y_col, random_value);
        }
    }
}

bool matrix_copy(const matrix *src, matrix *dest)
{
    if (src->rows != dest->rows || src->cols != dest->cols)
    {
        return false;
    }
    memcpy(dest->data, src->data, src->rows * src->cols * sizeof(double));
    return true;
}

// Operaciones con las matrices
bool matrix_add(const matrix *a, const matrix *b, matrix *result)
{
    if (a->rows != b->rows || a->cols != b->cols || a->rows != result->rows || a->cols != result->cols)
    {
        return false;
    }
    for (uint32_t x_row = 0; x_row < a->rows; x_row++)
    {
        for (uint32_t y_col = 0; y_col < a->cols; y_col++)
        {
            matrix_set(result, x_row, y_col, matrix_get(a, x_row, y_col) + matrix_get(b, x_row, y_col));
        }
    }
    return true;
}

bool matrix_sub(const matrix *a, const matrix *b, matrix *result)
{
    if (a->rows != b->rows || a->cols != b->cols || a->rows != result->rows || a->cols != result->cols)
    {
        return false;
    }
    for (uint32_t x_row = 0; x_row < a->rows; x_row++)
    {
        for (uint32_t y_col = 0; y_col < a->cols; y_col++)
        {
            matrix_set(result, x_row, y_col, matrix_get(a, x_row, y_col) - matrix_get(b, x_row, y_col));
        }
    }
    return true;
}
bool matrix_mult_scalar(const matrix *m, double scalar, matrix *result)
{
    if (m->rows != result->rows || m->cols != result->cols)
    {
        return false;
    }
    for (uint32_t x_row = 0; x_row < result->rows; x_row++)
    {
        for (uint32_t y_col = 0; y_col < result->cols; y_col++)
        {
            matrix_set(result, x_row, y_col, matrix_get(m, x_row, y_col) * scalar);
        }
    }
    return true;
}

bool matrix_mult(const matrix *a, const matrix *b, matrix *result)
{
    double suma_row_col = 0;

    if (a->cols != b->rows)
    {
        return false;
    }
    if (a->rows != result->rows || b->cols != result->cols)
    {
        return false;
    }

    for (uint32_t x_row = 0; x_row < a->rows; x_row++)
    {
        for (uint32_t y_col = 0; y_col < b->cols; y_col++)
        {
            suma_row_col = 0;

            for (uint32_t idx = 0; idx < a->cols; idx++)
            {
                suma_row_col += matrix_get(a, x_row, idx) * matrix_get(b, idx, y_col);
            }
            matrix_set(result, x_row, y_col, suma_row_col);
        }
    }
    return true;
}

bool matrix_transpose(const matrix *m, matrix *result)
{
    if (m->rows != result->cols || m->cols != result->rows)
    {
        return false;
    }
    for (uint32_t x_row = 0; x_row < m->rows; x_row++)
    {
        for (uint32_t y_col = 0; y_col < m->cols; y_col++)
        {
            matrix_set(result, y_col, x_row, matrix_get(m, x_row, y_col));
        }
    }
    return true;
}

// Utilidades
matrix *matrix_sum_rows(const matrix *m)
{ // Rows siempre igual a 1
    matrix *result = matrix_create(1, m->cols);
    double suma_columna;
    for (uint32_t y_col = 0; y_col < m->cols; y_col++)
    {
        suma_columna = 0;
        for (uint32_t x_row = 0; x_row < m->rows; x_row++)
        {
            suma_columna += matrix_get(m, x_row, y_col);
        }
        matrix_set(result, 0, y_col, suma_columna);
    }
    return result;
}

bool matrix_add_row_vector(matrix *m, const matrix *row_vector)
{
    if (row_vector->rows != 1 || m->cols != row_vector->cols)
    {
        return false;
    }
    for (uint32_t x_row = 0; x_row < m->rows; x_row++)
    {
        for (uint32_t y_col = 0; y_col < m->cols; y_col++)
        {
            matrix_set(m, x_row, y_col, matrix_get(m, x_row, y_col) + matrix_get(row_vector, 0, y_col));
        }
    }
    return true;
}

void matrix_apply_function(matrix *m, double (*func)(double))
{
    if (func == NULL)
    {
        return;
    }
    for (uint32_t x_row = 0; x_row < m->rows; x_row++)
    {
        for (uint32_t y_col = 0; y_col < m->cols; y_col++)
        {
            matrix_set(m, x_row, y_col, func(matrix_get(m, x_row, y_col)));
        }
    }
}

bool matrix_hadamard(const matrix *a, const matrix *b, matrix *result)
{
    if (a->rows != b->rows || a->cols != b->cols)
    {
        return false;
    }
    if (a->rows != result->rows || a->cols != result->cols)
    {
        return false;
    }
    for (uint32_t x_row = 0; x_row < a->rows; x_row++)
    {
        for (uint32_t y_col = 0; y_col < a->cols; y_col++)
        {
            matrix_set(result, x_row, y_col, matrix_get(a, x_row, y_col) * matrix_get(b, x_row, y_col));
        }
    }
    return true;
}

double matrix_sum(const matrix *m)
{
    double suma = 0;
    for (uint32_t x_row = 0; x_row < m->rows; x_row++)
    {
        for (uint32_t y_col = 0; y_col < m->cols; y_col++)
        {
            suma += matrix_get(m, x_row, y_col);
        }
    }
    return suma;
}
double matrix_mean(const matrix *m)
{
    return matrix_sum(m) / matrix_size(m);
}
void matrix_print(const matrix *m)
{
    for (uint32_t x_row = 0; x_row < m->rows; x_row++)
    {
        for (uint32_t y_col = 0; y_col < m->cols; y_col++)
        {
            printf("Elemento[%u,%u]: %f ", x_row, y_col, matrix_get(m, x_row, y_col));
        }
        printf("\n");
    }
}

// Auxiliares
uint32_t matrix_size(const matrix *m)
{
    return m->rows * m->cols;
}
// End of file