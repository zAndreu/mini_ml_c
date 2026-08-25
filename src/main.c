#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

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
// End of file