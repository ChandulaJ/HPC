#include <stdio.h>
#include <string.h>
#include <time.h>

#define MAX_LINE_LENGTH 1024
char inputFileLocation[200] = "/home/cj/HPC_data/Human_genome_preprocessed.fna";

void search(const char *line, const char *pattern) {
    int n = strlen(line);
    int m = strlen(pattern);
    int found = 0;

    for (int i = 0; i <= n - m; i++) {
        int j;
        for (j = 0; j < m; j++) {
            if (line[i + j] != pattern[j])
                break;
        }
        if (j == m) {
            printf("Pattern found at line %d, index %d\n", lineNumber, i);
            found = 1;
        }
    }
}

int main() {
    FILE *infile;
    char pattern[100];
    char line[MAX_LINE_LENGTH];

    printf("Enter pattern to search: ");
    scanf("%s", pattern);

    infile = fopen(filename, "r");
    if (infile == NULL) {
        perror("Error opening file");
        return 1;
    }

    clock_t start = clock();

    while (fgets(line, sizeof(line), infile)) {
        search(line, pattern);
    }

    clock_t end = clock();
    fclose(infile);

    double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for search: %.6f seconds\n", timeTaken);

    return 0;
}
