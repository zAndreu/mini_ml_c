#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "matrix.h"

#define MR 6
#define NR 32

#define KC 192
#define MC 192
#define NC 256

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

static inline void pack_A(const double *restrict src,
                          double *restrict dst,
                          uint32_t mr_eff,
                          uint32_t k_eff,
                          uint32_t stride_src)
{
    for (uint32_t i = 0; i < mr_eff; ++i)
    {
        memcpy(dst + (size_t)i * k_eff,
               src + (size_t)i * stride_src,
               (size_t)k_eff * sizeof(*dst));
    }
}

static inline void pack_B(const double *restrict src,
                          double *restrict dst,
                          uint32_t nr_eff,
                          uint32_t k_eff,
                          uint32_t stride_src)
{
    for (uint32_t k = 0; k < k_eff; ++k)
    {
        memcpy(dst + (size_t)k * NR,
               src + (size_t)k * stride_src,
               (size_t)nr_eff * sizeof(*dst));
    }
}

static inline void micro_mult(const double *restrict PackA,
                              const double *restrict PackB,
                              double *restrict C,
                              uint32_t k_eff,
                              uint32_t stride_C,
                              uint32_t mr_eff,
                              uint32_t nr_eff,
                              bool is_first_k)
{
    double acc[MR][NR];

    if (is_first_k)
    {
        for (uint32_t i = 0; i < mr_eff; ++i)
        {
            for (uint32_t j = 0; j < nr_eff; ++j)
            {
                acc[i][j] = 0.0;
            }
        }
    }
    else
    {
        for (uint32_t i = 0; i < mr_eff; ++i)
        {
            const double *restrict c_row =
                C + (size_t)i * stride_C;

            for (uint32_t j = 0; j < nr_eff; ++j)
            {
                acc[i][j] = c_row[j];
            }
        }
    }

    for (uint32_t i = 0; i < mr_eff; ++i)
    {
        const double *restrict a_row =
            PackA + (size_t)i * k_eff;

        double *restrict acc_row = acc[i];

        for (uint32_t k = 0; k < k_eff; ++k)
        {
            const double a_ik = a_row[k];

            const double *restrict b_row =
                PackB + (size_t)k * NR;

            for (uint32_t j = 0; j < nr_eff; ++j)
            {
                acc_row[j] += a_ik * b_row[j];
            }
        }
    }

    for (uint32_t i = 0; i < mr_eff; ++i)
    {
        double *restrict c_row =
            C + (size_t)i * stride_C;

        const double *restrict acc_row = acc[i];

        for (uint32_t j = 0; j < nr_eff; ++j)
        {
            c_row[j] = acc_row[j];
        }
    }
}

static bool multiply_block(const matrix *a,
                           const matrix *b,
                           matrix *result)
{
    const double *restrict A = a->data;
    const double *restrict B = b->data;
    double *restrict C = result->data;

    const uint32_t rows_A = a->rows;
    const uint32_t cols_A = a->cols;
    const uint32_t cols_B = b->cols;

#pragma omp parallel for collapse(2) schedule(static) if ((size_t)rows_A * cols_B > 65536u)
    for (uint32_t ii = 0; ii < rows_A; ii += MC)
    {
        for (uint32_t jj = 0; jj < cols_B; jj += NC)
        {
            double PackA[MR * KC] __attribute__((aligned(64)));
            double PackB[NC * KC] __attribute__((aligned(64)));

            const uint32_t rows_remaining = rows_A - ii;
            const uint32_t cols_remaining = cols_B - jj;

            const uint32_t row_max =
                (rows_remaining < MC) ? rows_A : ii + MC;

            const uint32_t col_max =
                (cols_remaining < NC) ? cols_B : jj + NC;

            for (uint32_t kk = 0; kk < cols_A; kk += KC)
            {
                const uint32_t k_remaining = cols_A - kk;

                const uint32_t k_eff =
                    (k_remaining < KC) ? k_remaining : KC;

                for (uint32_t j = jj; j < col_max; j += NR)
                {
                    const uint32_t nr_eff =
                        ((col_max - j) < NR)
                            ? (col_max - j)
                            : NR;

                    double *restrict packed_b_panel =
                        PackB + (size_t)(j - jj) * KC;

                    pack_B(B + (size_t)kk * cols_B + j,
                           packed_b_panel,
                           nr_eff,
                           k_eff,
                           cols_B);
                }

                for (uint32_t i = ii; i < row_max; i += MR)
                {
                    const uint32_t mr_eff =
                        ((row_max - i) < MR)
                            ? (row_max - i)
                            : MR;

                    pack_A(A + (size_t)i * cols_A + kk,
                           PackA,
                           mr_eff,
                           k_eff,
                           cols_A);

                    for (uint32_t j = jj; j < col_max; j += NR)
                    {
                        const uint32_t nr_eff =
                            ((col_max - j) < NR)
                                ? (col_max - j)
                                : NR;

                        const double *restrict packed_b_panel =
                            PackB + (size_t)(j - jj) * KC;

                        micro_mult(PackA,
                                   packed_b_panel,
                                   C + (size_t)i * cols_B + j,
                                   k_eff,
                                   cols_B,
                                   mr_eff,
                                   nr_eff,
                                   kk == 0);
                    }
                }
            }
        }
    }

    return true;
}

bool matrix_mult(const matrix *a,
                 const matrix *b,
                 matrix *result)
{
    if (a == NULL || b == NULL || result == NULL)
    {
        return false;
    }

    if ((a->rows != 0 && a->cols != 0 && a->data == NULL) ||
        (b->rows != 0 && b->cols != 0 && b->data == NULL) ||
        (result->rows != 0 &&
         result->cols != 0 &&
         result->data == NULL))
    {
        return false;
    }

    if (a->cols != b->rows ||
        result->rows != a->rows ||
        result->cols != b->cols)
    {
        return false;
    }

    if (result->data == a->data ||
        result->data == b->data)
    {
        return false;
    }

    if (result->rows == 0 || result->cols == 0)
    {
        return true;
    }

    if (a->cols == 0)
    {
        memset(result->data,
               0,
               (size_t)result->rows *
                   result->cols *
                   sizeof(*result->data));

        return true;
    }

    return multiply_block(a, b, result);
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
