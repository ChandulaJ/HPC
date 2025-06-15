#include <stdio.h>
#include <string.h>
#include <time.h>

#define MAX_LINE_LENGTH 1024
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
    char line[MAX_LINE_LENGTH];
    int totalFound = 0;

    printf("Enter pattern to search: ");
    scanf("%s", pattern);

    infile = fopen(inputFileLocation, "r");
    if (infile == NULL)
    {
        perror("Error opening file");
        return 1;
    }

    clock_t start = clock();
    printf("Searching for pattern: %s\n", pattern);
    printf("============================================================================\n");

    while (fgets(line, sizeof(line), infile))
    {
        totalFound += search(line, pattern);
    }
    printf("============================================================================\n");
    clock_t end = clock();
    fclose(infile);

    double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Total matches found: %d\n", totalFound);
    printf("Time taken for serial search: %.6f seconds\n", timeTaken);

    return 0;
}
