#include <stdio.h>
#include "matrix.h"

int main(void)
{
    matrix *m1 = matrix_create(8192, 8192);
    matrix *m2 = matrix_create(8192, 8192);
    matrix_fill_random(m1, -8192.0, 8192.0);
    matrix_fill_random(m2, -8192.0, 8192.0);

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