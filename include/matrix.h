#include <stdint.h>
#include <stdbool.h>

#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    uint32_t rows;
    uint32_t cols;
    double *data;
} matrix;

// Constructor y destructor
matrix *matrix_create(uint32_t rows, uint32_t cols);
void matrix_free(matrix *m);

// Getters y setters de elementos
double matrix_get(const matrix *m, uint32_t row, uint32_t col);
void matrix_set(matrix *m, uint32_t row, uint32_t col, double value);

void matrix_fill(matrix *m, double value);
void matrix_random_fill(matrix *m, double min, double max);
bool matrix_copy(const matrix *src, matrix *dest);

// Operaciones con las matrices
bool matrix_add(const matrix *a, const matrix *b, matrix *result);
bool matrix_sub(const matrix *a, const matrix *b, matrix *result);
bool matrix_mult_scalar(const matrix *m, double scalar, matrix *result);
bool matrix_mult(const matrix *a, const matrix *b, matrix *result);
bool matrix_transpose(const matrix *m, matrix *result);

// Utilidades
matrix *matrix_sum_rows(const matrix *m); // Rows siempre igual a 1
bool matrix_add_row_vector(matrix *m, const matrix *row_vector);
void matrix_apply_function(matrix *m, double (*func)(double));
bool matrix_hadamard(const matrix *a, const matrix *b, matrix *result);
double matrix_sum(const matrix *m);
double matrix_mean(const matrix *m);
void matrix_print(const matrix *m);

// Auxiliares
uint32_t matrix_size(const matrix *m);

#endif // MATRIX_H
// End of file