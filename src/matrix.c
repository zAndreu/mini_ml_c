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

#define TRANSPOSE_TILE 32

matrix *matrix_create(uint32_t rows, uint32_t cols)
{
    if (rows == 0 || cols == 0)
    {
        return NULL;
    }
    if ((size_t)rows > SIZE_MAX / (size_t)cols)
    {
        return NULL;
    }
    size_t count = (size_t)rows * (size_t)cols;
    if (count > SIZE_MAX / sizeof(double))
    {
        return NULL;
    }
    size_t size = count * sizeof(double);
    if (size > SIZE_MAX - 63)
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
    size_t aligned_size = (size + 63) & ~(size_t)63;

    m->data = (double *)aligned_alloc(64, aligned_size);
    if (m->data == NULL)
    {
        free(m);
        return NULL;
    }
    memset(m->data, 0, aligned_size);
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
    if (m == NULL || row >= m->rows || col >= m->cols)
    {
        return 0.0f;
    }
    return m->data[row * m->cols + col];
}

void matrix_set(matrix *m, uint32_t row, uint32_t col, double value)
{
    if (m == NULL || row >= m->rows || col >= m->cols)
    {
        return;
    }
    m->data[row * m->cols + col] = value;
}

void matrix_fill(matrix *m, double value)
{
    if (m == NULL)
    {
        return;
    }
    double *restrict data_ptr = m->data;
    const size_t total_elements = (size_t)m->rows * m->cols;

    for (size_t i = 0; i < total_elements; i++)
    {
        data_ptr[i] = value;
    }
}

void matrix_fill_random(matrix *m, double min, double max)
{
    if (m == NULL)
    {
        return;
    }
    double *restrict data_ptr = m->data;
    const size_t total_elements = (size_t)m->rows * m->cols;
    const double range = max - min;

    unsigned int seed = (unsigned int)(time(NULL) ^ (uintptr_t)&seed);

    for (size_t i = 0; i < total_elements; i++)
    {
        data_ptr[i] = min + ((double)rand_r(&seed) / (double)RAND_MAX) * range;
    }
}

bool matrix_copy(const matrix *src, matrix *dest)
{
    if (src == NULL || dest == NULL)
    {
        return false;
    }
    if (src->rows != dest->rows || src->cols != dest->cols)
    {
        return false;
    }
    if (src == dest || src->data == dest->data)
    {
        return true;
    }
    memcpy(dest->data, src->data, (size_t)src->rows * src->cols * sizeof(double));
    return true;
}

// Operaciones con las matrices
bool matrix_add(const matrix *a, const matrix *b, matrix *result)
{
    if (a == NULL || b == NULL || result == NULL)
    {
        return false;
    }
    if (a->rows != b->rows || a->cols != b->cols || a->rows != result->rows || a->cols != result->cols)
    {
        return false;
    }
    if (result->data == a->data || result->data == b->data)
    {
        return false;
    }
    const size_t total_elements = (size_t)a->rows * a->cols;
    const double *restrict pa = a->data;
    const double *restrict pb = b->data;
    double *restrict pr = result->data;
    for (size_t i = 0; i < total_elements; i++)
    {
        pr[i] = pa[i] + pb[i];
    }
    return true;
}

bool matrix_sub(const matrix *a, const matrix *b, matrix *result)
{
    if (a == NULL || b == NULL || result == NULL)
    {
        return false;
    }
    if (a->rows != b->rows || a->cols != b->cols || a->rows != result->rows || a->cols != result->cols)
    {
        return false;
    }
    if (result->data == a->data || result->data == b->data)
    {
        return false;
    }

    const size_t total_elements = (size_t)a->rows * a->cols;
    const double *restrict pa = a->data;
    const double *restrict pb = b->data;
    double *restrict pr = result->data;

    for (size_t i = 0; i < total_elements; i++)
    {
        pr[i] = pa[i] - pb[i];
    }
    return true;
}

bool matrix_hadamard(const matrix *a, const matrix *b, matrix *result)
{
    if (a == NULL || b == NULL || result == NULL)
    {
        return false;
    }
    if (a->rows != b->rows || a->cols != b->cols ||
        a->rows != result->rows || a->cols != result->cols)
    {
        return false;
    }
    if (result->data == a->data || result->data == b->data)
    {
        return false;
    }

    const size_t total_elements = (size_t)a->rows * a->cols;
    const double *restrict pa = a->data;
    const double *restrict pb = b->data;
    double *restrict pr = result->data;
    for (size_t i = 0; i < total_elements; i++)
    {
        pr[i] = pa[i] * pb[i];
    }
    return true;
}

bool matrix_mult_scalar(const matrix *m, double scalar, matrix *result)
{
    if (m == NULL || result == NULL)
    {
        return false;
    }
    if (m->rows != result->rows || m->cols != result->cols)
    {
        return false;
    }
    if (result->data == m->data)
    {
        return false;
    }

    const size_t total_elements = (size_t)m->rows * m->cols;
    const double *restrict src = m->data;
    double *restrict dst = result->data;

    for (size_t i = 0; i < total_elements; i++)
    {
        dst[i] = src[i] * scalar;
    }
    return true;
}

static inline void micro_mult(const double *restrict A, const double *restrict Bt, double *restrict C, uint32_t K_eff, uint32_t stride_A, uint32_t stride_Bt, uint32_t stride_C, uint32_t mr_eff, uint32_t nr_eff)
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
    double *restrict A = a->data;
    double *restrict Bt = bt->data;
    double *restrict C = result->data;
    memset(C, 0, result->rows * result->cols * sizeof(double));

#pragma omp parallel for schedule(static) // No me gusta depender de esto ya que lo cuento como ayuda externa a mi.
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
    const double *restrict src = m->data;
    double *restrict dst = result->data;
    const uint32_t rows = m->rows;
    const uint32_t cols = m->cols;

#pragma omp parallel for schedule(static)
    for (uint32_t ii = 0; ii < rows; ii += TRANSPOSE_TILE)
    {
        const uint32_t row_max = (ii + TRANSPOSE_TILE < rows) ? ii + TRANSPOSE_TILE : rows;

        for (uint32_t jj = 0; jj < cols; jj += TRANSPOSE_TILE)
        {
            const uint32_t col_max = (jj + TRANSPOSE_TILE < cols) ? jj + TRANSPOSE_TILE : cols;

            for (uint32_t i = ii; i < row_max; i++)
            {
                const double *src_row = &src[i * cols];

                for (uint32_t j = jj; j < col_max; j++)
                {
                    dst[j * rows + i] = src_row[j];
                }
            }
        }
    }
    return true;
}

bool matrix_transpose(const matrix *m, matrix *result)
{
    if (m == NULL || result == NULL)
    {
        return false;
    }
    if (m->rows != result->cols || m->cols != result->rows)
    {
        return false;
    }
    if (result->data == m->data)
    {
        return false;
    }
    return transpose_block(m, result);
}

// Utilidades
matrix *matrix_sum_rows(const matrix *m)
{
    if (m == NULL)
    {
        return NULL;
    }

    matrix *result = matrix_create(1, m->cols);
    if (!result)
    {
        return NULL;
    }

    double *restrict result_ptr = result->data;
    const double *restrict data_ptr = m->data;
    const uint32_t rows = m->rows;
    const uint32_t cols = m->cols;

    for (uint32_t r = 0; r < rows; r++)
    {
        const size_t offset = (size_t)r * cols;
        for (uint32_t c = 0; c < cols; c++)
        {
            result_ptr[c] += data_ptr[offset + c];
        }
    }

    return result;
}

bool matrix_add_row_vector(matrix *m, const matrix *row_vector)
{
    if (m == NULL || row_vector == NULL)
    {
        return false;
    }
    if (row_vector->rows != 1 || m->cols != row_vector->cols)
    {
        return false;
    }

    double *restrict data_ptr = m->data;
    const double *restrict row_vector_ptr = row_vector->data;
    const uint32_t rows = m->rows;
    const uint32_t cols = m->cols;

    for (uint32_t r = 0; r < rows; r++)
    {
        const size_t offset = (size_t)r * cols;
        for (uint32_t c = 0; c < cols; c++)
        {
            data_ptr[offset + c] += row_vector_ptr[c];
        }
    }
    return true;
}

void matrix_apply_function(matrix *m, double (*func)(double))
{
    if (m == NULL || func == NULL)
    {
        return;
    }
    double *restrict data_ptr = m->data;
    const size_t total_elements = (size_t)m->rows * m->cols;

    for (size_t i = 0; i < total_elements; i++)
    {
        data_ptr[i] = func(data_ptr[i]);
    }
}

double matrix_sum(const matrix *m)
{
    if (m == NULL)
    {
        return 0.0f;
    }

    double sum = 0.0f;
    const size_t total_elements = (size_t)m->rows * m->cols;
    const double *restrict data_ptr = m->data;

    for (size_t i = 0; i < total_elements; i++)
    {
        sum += data_ptr[i];
    }
    return sum;
}

double matrix_mean(const matrix *m)
{
    if (m == NULL)
    {
        return 0.0f;
    }
    size_t sz = matrix_size(m);
    if (sz == 0)
    {
        return 0.0f;
    }
    return matrix_sum(m) / (double)sz;
}

void matrix_print(const matrix *m)
{
    if (m == NULL)
    {
        return;
    }
    for (uint32_t x_row = 0; x_row < m->rows; x_row++)
    {
        for (uint32_t y_col = 0; y_col < m->cols; y_col++)
        {
            printf("%f ", matrix_get(m, x_row, y_col));
        }
        printf("\n");
    }
}

// Auxiliares
size_t matrix_size(const matrix *m)
{
    if (m == NULL)
    {
        return 0;
    }
    return (size_t)m->rows * m->cols;
}
// End of file
