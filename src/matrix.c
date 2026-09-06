#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "matrix.h"

static const uint16_t DBLOCK_L2 = 128;
static const uint16_t DKC_L1 = 192;
static const uint16_t DMR = 6;
static const uint16_t DNR = 8;

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
            double random_value = min + ((double)rand() / RAND_MAX) * (max - min);
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

static inline void micro_mult(const double *__restrict A, const double *__restrict Bt, double *__restrict C, uint32_t K_eff, uint32_t stride_A, uint32_t stride_Bt, uint32_t stride_C, uint32_t mr_eff, uint32_t nr_eff)
{
    double acc[DMR][DNR];
    memset(acc, 0, sizeof(acc));
    for (uint32_t i = 0; i < mr_eff; i++)
    {
        for (uint32_t j = 0; j < nr_eff; j++)
        {
            double sum = 0.0;
            for (uint32_t k = 0; k < K_eff; k++)
            {
                sum += A[i * stride_A + k] * Bt[j * stride_Bt + k];
            }

            acc[i][j] = sum;
        }
    }

    for (uint32_t i = 0; i < mr_eff; i++)
    {
        for (uint32_t j = 0; j < nr_eff; j++)
        {
            C[i * stride_C + j] += acc[i][j];
        }
    }
}

static bool multiply_block(const matrix *a, const matrix *bt, matrix *result)
{
    double *__restrict A = a->data;
    double *__restrict Bt = bt->data;
    double *__restrict C = result->data;

    memset(C, 0, result->rows * result->cols * sizeof(double));

    // #pragma omp parallel for schedule(static) // No me gusta depender de esto ya que lo cuento como ayuda externa a mi.
    for (uint32_t ii = 0; ii < a->rows; ii += DBLOCK_L2)
    {
        uint32_t row_max = (ii + DBLOCK_L2 < a->rows) ? ii + DBLOCK_L2 : a->rows;
        for (uint32_t jj = 0; jj < bt->rows; jj += DBLOCK_L2)
        {
            uint32_t col_max = (jj + DBLOCK_L2 < bt->rows) ? jj + DBLOCK_L2 : bt->rows;
            for (uint32_t kk = 0; kk < a->cols; kk += DKC_L1)
            {
                uint32_t k_max = (kk + DKC_L1 < a->cols) ? kk + DKC_L1 : a->cols;
                uint32_t k_eff = k_max - kk;
                for (uint32_t i = ii; i < row_max; i += DMR)
                {
                    uint32_t mr_eff = (i + DMR < row_max) ? DMR : (row_max - i);
                    for (uint32_t j = jj; j < col_max; j += DNR)
                    {
                        uint32_t nr_eff = (j + DNR < col_max) ? DNR : (col_max - j);
                        micro_mult(&A[i * a->cols + kk], &Bt[j * bt->cols + kk], &C[i * result->cols + j], k_eff, a->cols, bt->cols, result->cols, mr_eff, nr_eff);
                    }
                }
            }
        }
    }

    return true;
}

bool matrix_mult(const matrix *a, const matrix *b, matrix *result)
{
    if (a->cols != b->rows)
    {
        return false;
    }
    if (a->rows != result->rows || b->cols != result->cols)
    {
        return false;
    }
    matrix *b_transposed = matrix_create(b->cols, b->rows);
    matrix_transpose(b, b_transposed);
    if (!multiply_block(a, b_transposed, result))
    {
        return false;
    }
    matrix_free(b_transposed);
    return true;
}

static bool transpose_block(const matrix *m, matrix *result)
{
    double *__restrict src = m->data;
    double *__restrict dst = result->data;

    // #pragma omp parallel for schedule(static) // No me gusta depender de esto ya que lo cuento como ayuda externa a mi.
    for (uint32_t ii = 0; ii < m->rows; ii += DBLOCK_L2)
    {
        uint32_t row_max = (ii + DBLOCK_L2 < m->rows) ? ii + DBLOCK_L2 : m->rows;
        for (uint32_t jj = 0; jj < m->cols; jj += DBLOCK_L2)
        {
            uint32_t col_max = (jj + DBLOCK_L2 < m->cols) ? jj + DBLOCK_L2 : m->cols;
            for (uint32_t i = ii; i < row_max; i++)
            {
                uint32_t row_offset = i * m->cols;
                for (uint32_t j = jj; j < col_max; j++)
                {
                    dst[j * m->rows + i] = src[row_offset + j];
                }
            }
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
    if (!transpose_block(m, result))
    {
        return false;
    }
    return true;
}

// Utilidades
matrix *matrix_sum_rows(const matrix *m) // Rows siempre igual a 1
{
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