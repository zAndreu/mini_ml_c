#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../include/matrix.h"

matrix *matrix_create(uint32_t rows, uint32_t cols) {
    matrix *m = (matrix *)malloc(sizeof(matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (double *)malloc(rows * cols * sizeof(double));
    return m;
}

void matrix_free(matrix *m) {
    free(m->data);
    free(m);
}

double matrix_get(const matrix *m, uint32_t row, uint32_t col) {
    return m->data[row * m->cols + col];
}

void matrix_set(matrix *m, uint32_t row, uint32_t col, double value) {
    m->data[row * m->cols + col] = value;
}

void matrix_fill(matrix *m, double value) {
    for(uint32_t x_row = 0; x_row < m->rows; x_row++) {
        for(uint32_t y_col = 0; y_col < m->cols; y_col++) {
            matrix_set(m, x_row, y_col, value);
        }
    }
}

void matrix_random_fill(matrix *m, double min, double max) {
    for(uint32_t x_row = 0; x_row < m->rows; x_row++) {
        for(uint32_t y_col = 0; y_col < m->cols; y_col++) {
            double random_value = ((double)rand() / RAND_MAX) * (max - min) + min;
            matrix_set(m, x_row, y_col, random_value);
        }
    }
    
}

bool matrix_copy(const matrix *src, matrix *dest) {
    if (src->rows != dest->rows || src->cols != dest->cols) {
        return false;
    }
    memcpy(dest->data, src->data, src->rows * src->cols * sizeof(double));
    return true;
}

// Operaciones con las matrices
bool matrix_add(const matrix *a, const matrix *b, matrix *result) {
    if (a->rows != b->rows || a->cols != b->cols || a->rows != result->rows || a->cols != result->cols) {
        return false;
    }
    for (uint32_t x_row = 0; x_row < a->rows; x_row++) {
        for (uint32_t y_col = 0; y_col < a->cols; y_col++) {
            matrix_set(result, x_row, y_col, matrix_get(a, x_row, y_col) + matrix_get(b, x_row, y_col));
        }
    }
    return true;
}

bool matrix_sub(const matrix *a, const matrix *b, matrix *result) {
    if (a->rows != b->rows || a->cols != b->cols || a->rows != result->rows || a->cols != result->cols) {
        return false;
    }
    for (uint32_t x_row = 0; x_row < a->rows; x_row++) {
        for (uint32_t y_col = 0; y_col < a->cols; y_col++) {
            matrix_set(result, x_row, y_col, matrix_get(a, x_row, y_col) - matrix_get(b, x_row, y_col));
        }
    }
    return true;
}
bool matrix_mult_scalar(const matrix *m, double scalar, matrix *result) {
    for (uint32_t x_row = 0; x_row < result->rows; x_row++) {
        for (uint32_t y_col = 0; y_col < result->cols; y_col++) {
            matrix_set(result, x_row, y_col, matrix_get(m, x_row, y_col) * scalar);
        }
    }
    return true;
}

bool matrix_mult(const matrix *a, const matrix *b, matrix *result) {
    double suma_row_col = 0;

    if (a->cols != b->rows) {
        return false;
    }
    if (a->rows != result->rows || b->cols != result->cols) {
        return false;
    }

    for (uint32_t x_row = 0; x_row < a->rows; x_row++) {
        for (uint32_t y_col = 0; y_col < b->cols; y_col++) {
            suma_row_col = 0;

            for (uint32_t idx = 0; idx < a->cols; idx++) {
                suma_row_col += matrix_get(a, x_row, idx) * matrix_get(b, idx, y_col);
            }
            matrix_set(result, x_row, y_col, suma_row_col);
        }
    }
    return true; 
}

bool matrix_transpose(const matrix *m, matrix *result) {
    if (m->rows != result->cols || m->cols != result->rows) {
        return false;
    }
    for (uint32_t x_row = 0; x_row < m->rows; x_row++) {
        for (uint32_t y_col = 0; y_col < m->cols; y_col++) {
            matrix_set(result, x_row, y_col, matrix_get(m, y_col, x_row));
        }
    }
    return true;
}

// Utilidades
matrix *matrix_sum_rows(const matrix *m) { // Rows siempre igual a 1 
    matrix *result = matrix_create(1, m->cols);
    double suma_columna;
    for (uint32_t y_col = 0; y_col < m->cols; y_col++) {
        suma_columna = 0;
        for (uint32_t x_row = 0; x_row < m->rows; x_row++) {
            suma_columna += matrix_get(m, x_row, y_col);
        }
        matrix_set(result, 0, y_col, suma_columna);
    }
    return result;
}

bool matrix_add_row_vector(matrix *m, const matrix *row_vector) {
    if (row_vector->rows != 1 || m->cols != row_vector->cols) {
        return false;
    }
    for (uint32_t x_row = 0; x_row < m->rows; x_row++) {
        for (uint32_t y_col = 0; y_col < m->cols; y_col++) {
            matrix_set(m, x_row, y_col, matrix_get(m, x_row, y_col) + matrix_get(row_vector, 0, y_col));
        }
    }
    return true;
}

void matrix_apply_function(matrix *m, double (*func)(double)) {
    for (int x_row = 0; x_row < m->rows; x_row++) {
        for (int y_col = 0; y_col < m->cols; y_col++) {
            matrix_set(m, x_row, y_col, func(matrix_get(m, x_row, y_col)));
        }
    }
}

bool matrix_hadamard(const matrix *a, const matrix *b, matrix *result) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return false;
    }
    if (a->rows != result->rows || a->cols != result->cols) {
        return false;
    }
    for (uint32_t x_row = 0; x_row < a->rows; x_row++) {
        for (uint32_t y_col = 0; y_col < a->cols; y_col++) {
            matrix_set(result, x_row, y_col, matrix_get(a, x_row, y_col) * matrix_get(b, x_row, y_col));
        }
    }
    return true;
}

double matrix_sum(const matrix *m) {
    double suma = 0;
    for (uint32_t x_row = 0; x_row < m->rows; x_row++) {
        for (uint32_t y_col = 0; y_col < m->cols; y_col++) {
            suma += matrix_get(m, x_row, y_col);
        }
    }
    return suma;
}
double matrix_mean(const matrix *m) {
    return matrix_sum(m) / matrix_size(m);
}
void matrix_print(const matrix *m) {
    for (uint32_t x_row = 0; x_row < m->rows; x_row++) {
        for (uint32_t y_col = 0; y_col < m->cols; y_col++) {
            printf("Elemento[%d,%d]: %lf ", x_row, y_col, matrix_get(m, x_row, y_col));
        }
        printf("\n");
    }
}

// Auxiliares
uint32_t matrix_size(const matrix *m) {
    return m->rows * m->cols;
}

int main() {
    matrix *a = matrix_create(2, 3);
    matrix *b = matrix_create(2, 3);
    matrix *result = matrix_create(2, 3);
    matrix *product_a = matrix_create(2, 2);
    matrix *product_b = matrix_create(2, 2);
    matrix *product = matrix_create(2, 2);
    matrix *transpose = matrix_create(3, 2);
    matrix *row_vector = matrix_create(1, 3);
    matrix *copy = matrix_create(2, 3);

    matrix_fill(a, 2.0);
    matrix_random_fill(b, 0.0, 1.0);
    matrix_set(a, 0, 1, 5.0);
    printf("a[0,1] = %lf\n", matrix_get(a, 0, 1));

    matrix_copy(a, copy);
    matrix_add(a, b, result);
    printf("a + b:\n");
    matrix_print(result);

    matrix_sub(a, b, result);
    printf("a - b:\n");
    matrix_print(result);

    matrix_mult_scalar(a, 3.0, result);
    printf("a * escalar:\n");
    matrix_print(result);

    matrix_fill(product_a, 2.0);
    matrix_fill(product_b, 3.0);
    matrix_mult(product_a, product_b, product);
    printf("producto:\n");
    matrix_print(product);

    matrix_transpose(a, transpose);
    printf("transpuesta de a:\n");
    matrix_print(transpose);

    matrix_fill(row_vector, 1.0);
    matrix_add_row_vector(a, row_vector);
    printf("a con vector fila sumado:\n");
    matrix_print(a);

    printf("suma = %lf, media = %lf, tamano = %u\n",
           matrix_sum(a), matrix_mean(a), matrix_size(a));

    matrix_free(a);
    matrix_free(b);
    matrix_free(result);
    matrix_free(product_a);
    matrix_free(product_b);
    matrix_free(product);
    matrix_free(transpose);
    matrix_free(row_vector);
    matrix_free(copy);
    return 0;
}