#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <limits.h>

#define MAX_LINE_LENGTH 200
#define MAX_LINES 50000000
#define NO_OF_CHARS 256

char inputFileLocation[200] = "/home/cj/HPC_data/Human_genome_preprocessed.fna";

// Utility function to get maximum of two integers
int max(int a, int b) { 
    return (a > b) ? a : b; 
}

// The preprocessing function for Boyer Moore's bad character heuristic
void badCharHeuristic(const char* str, int size, int badchar[NO_OF_CHARS])
{
    int i;
    // Initialize all occurrences as -1
    for (i = 0; i < NO_OF_CHARS; i++)
        badchar[i] = -1;
    
    // Fill the actual value of last occurrence of a character
    for (i = 0; i < size; i++)
        badchar[(int)str[i]] = i;
}

// Boyer-Moore pattern searching function with match counting and reporting
int boyerMooreSearch(const char *line, const char *pattern)
{
    int m = strlen(pattern);
    int n = strlen(line);
    int badchar[NO_OF_CHARS];
    int foundCount = 0;
    
    // Fill the bad character array by calling the preprocessing function
    badCharHeuristic(pattern, m, badchar);
    
    int s = 0; // s is shift of the pattern with respect to text
    while (s <= (n - m)) {
        int j = m - 1;
        
        // Keep reducing index j of pattern while characters match
        while (j >= 0 && pattern[j] == line[s + j])
            j--;
        
        // If the pattern is present at current shift
        if (j < 0) {
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
            
            // Shift the pattern so that the next character in text aligns 
            // with the last occurrence of it in pattern
            s += (s + m < n) ? m - badchar[line[s + m]] : 1;
        }
        else {
            // Shift the pattern so that the bad character in text aligns 
            // with the last occurrence of it in pattern
            s += max(1, j - badchar[line[s + j]]);
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
    char searchMethod;

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

    printf("Searching for pattern: %s using Boyer-Moore Algorithm\n", pattern);

    printf("============================================================================\n");

    clock_t start = clock();

    // Choose search algorithm based on user input

    for (int i = 0; i < lineCount; i++)
    {
        totalFound += boyerMooreSearch(lines[i], pattern);
    }
    
    clock_t end = clock();

    printf("============================================================================\n");

    double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Total matches found: %d\n", totalFound);

    printf("Time taken for Boyer-Moore search: %.6f seconds\n", timeTaken);

    // Cleanup
    for (int i = 0; i < lineCount; i++)
        free(lines[i]);
    free(lines);

    return 0;
}