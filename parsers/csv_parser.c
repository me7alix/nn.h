#include <stdio.h>
#include <string.h>
#include "../nn.h"

void count_rows_cols(FILE* file, size_t* rows, size_t* cols, const char *del) {
    char line[1 << 16];
    *rows = 0;
    *cols = 0;

    if (fgets(line, sizeof(line), file)) {
        (*rows)++;
        char* token = strtok(line, del);
        while (token) {
            (*cols)++;
            token = strtok(NULL, del);
        }
    }

    while (fgets(line, sizeof(line), file)) {
        (*rows)++;
    }
    rewind(file);
}

Mat parse_csv_to_mat(const char* filename, const char *del) {
    FILE* file = fopen(filename, "r");

    size_t rows, cols;
    count_rows_cols(file, &rows, &cols, del);

    Mat mat = mat_alloc(rows, cols);

    char line[1 << 16];
    size_t row = 0;

    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, del);
        for (size_t col = 0; col < cols && token; col++) {
            MAT_AT(mat, row, col) = strtof(token, NULL);
            token = strtok(NULL, del);
        }
        row++;
    }

    fclose(file);
    return mat;
}
