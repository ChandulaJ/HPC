#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#define MAX_LINE_LENGTH 1024
#define MAX_LINES 100000000
char inputFileLocation[200] = "/home/cj/HPC_data/Human_genome_preprocessed.fna";

int search(const char *line, const char *pattern)
{
    int lineSize = strlen(line);
    int pattSize = strlen(pattern);
    int foundCount = 0;

    for (int i = 0; i <= lineSize - pattSize; i++)
    {
        int j;
        for (j = 0; j < pattSize; j++)
        {
            if (line[i + j] != pattern[j])
                break;
        }
        if (j == pattSize)
        {
            char chromo[100], lineNum[100];

            // Find the position of '_' and '|'
            char *underscore = strchr(line, '_');
            char *pipe = strchr(line, '|');

            if (underscore && pipe && underscore < pipe)
            {
                // Copy chromo (from beginning to underscore)
                size_t len_chromo = underscore - line;
                strncpy(chromo, line, len_chromo);
                chromo[len_chromo] = '\0';

                // Copy lineNum (from underscore+1 to pipe)
                size_t len_lineNum = pipe - underscore - 1;
                strncpy(lineNum, underscore + 1, len_lineNum);
                lineNum[len_lineNum] = '\0';

                printf("Pattern found at chromosome %s, at line %s\n", chromo, lineNum);
                foundCount++;
            }
        }
    }
    return foundCount;
}

int main()
{
    FILE *infile;
    char pattern[100];
    char **lines = malloc(MAX_LINES * sizeof(char *));
    char buffer[MAX_LINE_LENGTH];
    int lineCount = 0, totalFound = 0;

    printf("Enter pattern to search: ");
    scanf("%s", pattern);

    infile = fopen(inputFileLocation, "r");
    if (infile == NULL)
    {
        perror("Error opening file");
        return 1;
    }
    printf("----------------------------------------------------------------\n");
    printf("Loading file to memory...\n");


    // Load the entire file into memory
    while (fgets(buffer, sizeof(buffer), infile) && lineCount < MAX_LINES)
    {
        lines[lineCount] = strdup(buffer);  // Allocate and copy line
        if (lines[lineCount] == NULL)
        {
            fprintf(stderr, "Memory allocation failed at line %d\n", lineCount);
            return 1;
        }
        lineCount++;
    }
    fclose(infile);

    printf("File loaded to memory successfully\n");
    printf("----------------------------------------------------------------\n");

    printf("Searching for pattern: %s\n", pattern);
    printf("============================================================================\n");

    clock_t start = clock();

    for (int i = 0; i < lineCount; i++)
    {
        totalFound += search(lines[i], pattern);
    }
    
    clock_t end = clock();

    printf("============================================================================\n");

    double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Total matches found: %d\n", totalFound);
    printf("Time taken for serial search: %.6f seconds\n", timeTaken);

    // Cleanup
    for (int i = 0; i < lineCount; i++)
        free(lines[i]);
    free(lines);

    return 0;
}