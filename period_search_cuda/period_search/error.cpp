#include "stdafx.h"
#include <cstdio>
#include <cstdlib>

// NOTE: Example usssage
//
//int main() {
//    const char* input = "123abc";
//    int number;
//    char buffer[100];
//
//    // Attempt to convert input string to an integer
//    int no_conversions = sscanf(input, "%d", &number);
//
//    if (no_conversions != 1) {
//        // Call ErrorFunction if conversion fails
//        strncpy(buffer, input, sizeof(buffer));
//        buffer[sizeof(buffer) - 1] = '\0'; // Ensure null-terminated string
//        ErrorFunction(buffer, no_conversions);
//    }
//
//    // If conversion succeeds, continue processing
//    printf("Conversion successful: %d\n", number);
//    return 0;
//}

/**
 * The ErrorFunction is designed to print an error message to stderr and then terminate the program if an error occurs.
 * @param buffer 
 * @param no_conversions 
 */
void ErrorFunction(const char* buffer, int no_conversions) {
    fprintf(stderr, "An error occurred. You entered:\n%s\n", buffer);
    fprintf(stderr, "%d successful conversions", no_conversions);
    exit(EXIT_FAILURE);
}