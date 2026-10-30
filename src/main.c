#include <stdio.h>
#include "matrix.h"

int main(void)
{
    matrix *m1 = matrix_create(2048, 2048);
    matrix *m2 = matrix_create(2048, 2048);
    matrix_fill_random(m1, -2048.0, 2048.0);
    matrix_fill_random(m2, -2048.0, 2048.0);

    matrix *result = matrix_create(m1->rows, m2->cols);
    if (matrix_mult(m1, m2, result))
    {
        printf("Matrix multiplication successful.\n");
    }
    matrix_free(m1);
    matrix_free(m2);
    matrix_free(result);
    return 0;
}
// End of file