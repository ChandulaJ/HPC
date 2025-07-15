#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#define MAX_LINE_LENGTH 200
#define MAX_LINES 50000000

char inputFileLocation[200] = "/home/cj/HPC_data/Human_genome_preprocessed.fna";

// KMP preprocessing function to build the failure function (LPS array)
void computeLPSArray(const char* pattern, int m, int* lps)
{
    int len = 0;  // length of the previous longest prefix suffix
    lps[0] = 0;   // lps[0] is always 0
    int i = 1;
    
    // Calculate lps[i] for i = 1 to m-1
    while (i < m) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
}

// KMP pattern searching function with match counting and reporting
int kmpSearch(const char *line, const char *pattern)
{
    int m = strlen(pattern);
    int n = strlen(line);
    int foundCount = 0;
    
    // Create LPS array for the pattern
    int* lps = (int*)malloc(m * sizeof(int));
    computeLPSArray(pattern, m, lps);
    
    int i = 0; // index for line
    int j = 0; // index for pattern
    
    while (i < n) {
        if (pattern[j] == line[i]) {
            i++;
            j++;
        }
        
        if (j == m) {
            // Pattern found - extract chromosome and line information
            char chromo[100], lineNum[100];
            
            // Find the position of '_' and '|'
            char *underscore = strchr(line, '_');
            char *pipe = strchr(line, '|');
            
            if (underscore && pipe && underscore < pipe) {
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
            
            j = lps[j - 1]; // Get the next position to check
        } else if (i < n && pattern[j] != line[i]) {
            // Mismatch after j matches
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
    
    free(lps);
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
    printf("Using KMP (Knuth-Morris-Pratt) Algorithm\n");
    printf("============================================================================\n");

    clock_t start = clock();

    // Use KMP algorithm for all searches
    for (int i = 0; i < lineCount; i++)
    {
        totalFound += kmpSearch(lines[i], pattern);
    }
    
    clock_t end = clock();

    printf("============================================================================\n");

    double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Total matches found: %d\n", totalFound);
    printf("Time taken for KMP search: %.6f seconds\n", timeTaken);

    // Cleanup
    for (int i = 0; i < lineCount; i++)
        free(lines[i]);
    free(lines);

    return 0;
}