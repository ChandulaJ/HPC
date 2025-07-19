#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_LINE 1024

int main() {
    char inputFileLocation[200];
    char outputFileLocation[200];

    printf("Enter input file location: ");
    scanf("%s", inputFileLocation);
    
    printf("Enter output file location: ");
    scanf("%s", outputFileLocation);

    if(inputFileLocation[0] == '\0') {
        strcpy(inputFileLocation, "/home/cj/HPC_data/Human_genome.fna");

    }
    if(outputFileLocation[0] == '\0') {
        strcpy(outputFileLocation, "/home/cj/HPC_data/Human_genome_preprocessed.fna");
    }

    FILE *infile = fopen(inputFileLocation, "r");
    FILE *outfile = fopen(outputFileLocation, "w");

    if (!infile || !outfile) {
        perror("File open error");
        return 1;
    }

    char line[MAX_LINE];
    char chromosome[11] = "";
    int line_num = 0;

    while (fgets(line, sizeof(line), infile)) {
        // Remove newline
        line[strcspn(line, "\r\n")] = 0;

        //when it is a chromosome header
        if (line[0] == '>') {
            printf("%s\n", line);
            strncpy(chromosome, line + 1, sizeof(chromosome));
            chromosome[sizeof(chromosome)-1] = '\0';  // Safe null-terminate
            line_num = 0;  // Reset line count
        } else if (strlen(line) > 0) {
            line_num++;
            // Format line number as 10-digit
            fprintf(outfile, "%s_%010d|%s\n", chromosome, line_num, line);
        }
    }

    fclose(infile);
    fclose(outfile);
    printf("Preprocessing complete. Output written to output.fna\n");

    return 0;
}
