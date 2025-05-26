#include <stdio.h>
#include <string.h>
#include <time.h>

#define MAX_LINE_LENGTH 1024

void search(const char *line, const char *pattern, int lineNumber) {
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
    FILE *fp;
    char filename[100], pattern[100];
    char line[MAX_LINE_LENGTH];
    int lineNumber = 0;

    printf("Enter file name: ");
    scanf("%s", filename);
    printf("Enter pattern to search: ");
    scanf("%s", pattern);

    fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Error opening file");
        return 1;
    }

    clock_t start = clock();

    while (fgets(line, sizeof(line), fp)) {
        lineNumber++;
        search(line, pattern, lineNumber);
    }

    clock_t end = clock();
    fclose(fp);

    double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for search: %.6f seconds\n", timeTaken);

    return 0;
}
