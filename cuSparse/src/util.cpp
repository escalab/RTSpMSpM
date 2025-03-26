#include <vector>

// ============================================================================
// Helper Functions

/**
 * Load in coo from data file
 */
void cooFromFile( const std::string& filePath, int** rowArr, int** colArr, 
                  float** valArr, int* rowSize, int* colSize, 
                  uint64_t* arrSize)
{
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file: " << filePath << std::endl;
        return;
    }
    std::string line;
    // Skip header lines here
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Read matrix metadata (num_rows, num_cols, nnz)
    int rows, cols;
    uint64_t nnz;
    std::istringstream iss(line);
    iss >> rows >> cols >> nnz;
    
    *rowSize = rows;
    *colSize = cols;
    *arrSize = nnz;

    // Allocate memory for COO arrays based on arrSize (number of non-zero elements)
    *rowArr = new int[*arrSize];
    *colArr = new int[*arrSize];
    *valArr = new float[*arrSize];

    // Read COO data (rows, columns, values)
    for (uint64_t i = 0; i < *arrSize; ++i) {
        file >> (*rowArr)[i] >> (*colArr)[i] >> (*valArr)[i];
    }
    file.close();
}

/**
 * Print coo to output file
 */
void printCooToFile( const std::string& filePath, const int* rowArr, 
                     const int* colArr, const float* valArr, const int rowSize,
                     const int colSize, uint64_t arrSize)
{
    // Open File
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return;
    }

    // MTX header
    outFile << "%%MatrixMarket matrix coordinate real general\n";
    // Dimensions and non-zero count
    outFile << rowSize << " " << colSize << " " << arrSize << "\n";

    for (uint64_t i = 0; i < arrSize; ++i){
        outFile << rowArr[i] << " " << colArr[i] << " " << valArr[i] << "\n";
    }
    outFile.close();
}

void coo_to_csr(const int* cooRow, uint64_t nnz, int num_rows, int* csrRowPtr) 
{
    // Initialize csrRowPtr array with ones for 1-based indexing
    std::fill(csrRowPtr, csrRowPtr + num_rows + 1, 1);

    // Step 1: Count occurrences of each row index in cooRow
    for (uint64_t i = 0; i < nnz; ++i) {
        csrRowPtr[cooRow[i] + 1]++;
    }

    // Step 2: Accumulate the counts to get the row pointers, starting from index 1
    for (uint64_t i = 1; i <= num_rows; ++i) {
        csrRowPtr[i] += csrRowPtr[i - 1];
    }
}

void csr_to_coo(const int* csrRowPtr, int num_rows, uint64_t nnz, int* cooRow, 
                bool is_one_based) 
{
    // Iterate over each row in CSR format
    for (uint64_t row = 0; row < num_rows; ++row) {
        uint64_t start = csrRowPtr[row] - (is_one_based ? 1 : 0);
        uint64_t end = csrRowPtr[row + 1] - (is_one_based ? 1 : 0);

        // Assign row index to COO format for each non-zero element in this row
        for (uint64_t i = start; i < end; ++i) {
            cooRow[i] = is_one_based ? row + 1 : row;
        }
    }
}

