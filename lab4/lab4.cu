#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#endif

using namespace std;

using Matrix = vector<int>;

struct CudaTimes {
    double total_seconds = 0.0;
    float kernel_ms = 0.0f;
};

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = (call);                                             \
        if (error != cudaSuccess) {                                             \
            throw runtime_error(string("CUDA error: ") + cudaGetErrorString(error)); \
        }                                                                      \
    } while (0)

__global__ void matrix_multiply_kernel(
    const int* A,
    const int* B,
    int* C,
    int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;

        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }

        C[row * n + col] = sum;
    }
}

Matrix read_matrix(const string& filename, int& rows, int& cols)
{
    ifstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    file >> rows >> cols;

    if (!file || rows <= 0 || cols <= 0) {
        throw runtime_error("Invalid matrix header in file: " + filename);
    }

    Matrix matrix(static_cast<size_t>(rows) * static_cast<size_t>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (!(file >> matrix[i * cols + j])) {
                throw runtime_error("Not enough matrix elements in file: " + filename);
            }
        }
    }

    return matrix;
}

void write_matrix(const string& filename, const Matrix& matrix, int rows, int cols)
{
    ofstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Cannot open file for writing: " + filename);
    }

    file << rows << ' ' << cols << '\n';

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix[i * cols + j];

            if (j + 1 < cols) {
                file << ' ';
            }
        }

        file << '\n';
    }
}

bool run_python_check()
{
#ifdef _WIN32
    // eqw.py возвращает 1, если результат правильный.
    int result = system("python eqw.py");
    return result == 1;
#else
    int result = system("python3 eqw.py");

    if (WIFEXITED(result)) {
        return WEXITSTATUS(result) == 1;
    }

    return false;
#endif
}

string get_cuda_device_name()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    return properties.name;
}

void write_result_file(
    const string& filename,
    const Matrix& matrix,
    int n,
    const CudaTimes& times,
    bool is_correct,
    const string& device_name
) {
    ofstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Cannot open file for writing: " + filename);
    }

    long long operations = (2LL * n - 1) * n * n;

    file << "Matrix size: " << n << " x " << n << '\n';
    file << "Technology: CUDA\n";
    file << "GPU: " << device_name << '\n';
    file << "Total CUDA time: " << times.total_seconds << " sec\n";
    file << "Total CUDA time: " << times.total_seconds * 1000.0 << " ms\n";
    file << "Kernel time: " << times.kernel_ms << " ms\n";
    file << "Number of operations: " << operations << '\n';
    file << "Correctly: " << is_correct << "\n\n";

    file << "Result matrix:\n";

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << matrix[i * n + j];

            if (j + 1 < n) {
                file << ' ';
            }
        }

        file << '\n';
    }
}

Matrix matrix_multiply_cuda(
    const Matrix& A,
    const Matrix& B,
    int n,
    CudaTimes& times
) {
    size_t element_count = static_cast<size_t>(n) * static_cast<size_t>(n);
    size_t bytes = element_count * sizeof(int);

    Matrix C(element_count, 0);

    int* d_A = nullptr;
    int* d_B = nullptr;
    int* d_C = nullptr;

    cudaEvent_t kernel_start{};
    cudaEvent_t kernel_stop{};

    auto total_start = chrono::high_resolution_clock::now();

    try {
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));

        CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventCreate(&kernel_start));
        CUDA_CHECK(cudaEventCreate(&kernel_stop));

        dim3 block_size(32, 16);
        dim3 grid_size(
            (n + block_size.x - 1) / block_size.x,
            (n + block_size.y - 1) / block_size.y
        );

        CUDA_CHECK(cudaEventRecord(kernel_start));

        matrix_multiply_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));
        CUDA_CHECK(cudaEventElapsedTime(&times.kernel_ms, kernel_start, kernel_stop));

        CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (...) {
        if (kernel_start) {
            cudaEventDestroy(kernel_start);
        }
        if (kernel_stop) {
            cudaEventDestroy(kernel_stop);
        }
        if (d_A) {
            cudaFree(d_A);
        }
        if (d_B) {
            cudaFree(d_B);
        }
        if (d_C) {
            cudaFree(d_C);
        }
        throw;
    }

    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    auto total_end = chrono::high_resolution_clock::now();
    times.total_seconds = chrono::duration<double>(total_end - total_start).count();

    return C;
}

int main()
{
    try {
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));

        if (device_count <= 0) {
            throw runtime_error("CUDA device was not found");
        }

        CUDA_CHECK(cudaSetDevice(0));

        int rows_A = 0;
        int cols_A = 0;
        int rows_B = 0;
        int cols_B = 0;

        Matrix A = read_matrix("matrixA.txt", rows_A, cols_A);
        Matrix B = read_matrix("matrixB.txt", rows_B, cols_B);

        if (rows_A != cols_A || rows_B != cols_B) {
            throw runtime_error("Input matrices must be square");
        }

        if (rows_A != rows_B) {
            throw runtime_error("Input matrices must have the same size");
        }

        int n = rows_A;

        CudaTimes times;
        Matrix C = matrix_multiply_cuda(A, B, n, times);

        write_matrix("matrixC.txt", C, n, n);

        bool is_correct = run_python_check();
        string device_name = get_cuda_device_name();

        write_result_file("matrixAB.txt", C, n, times, is_correct, device_name);

        cout << "Matrix size: " << n << " x " << n << endl;
        cout << "Technology: CUDA" << endl;
        cout << "GPU: " << device_name << endl;
        cout << "Total CUDA time: " << times.total_seconds << " sec" << endl;
        cout << "Kernel time: " << times.kernel_ms << " ms" << endl;
        cout << "Correctly: " << is_correct << endl;
    } catch (const exception& error) {
        cerr << "Error: " << error.what() << endl;
        return 1;
    }

    return 0;
}
