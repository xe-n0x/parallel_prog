#include <mpi.h>

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

using Matrix = vector<vector<int>>;

const int ROOT = 0;
const int TAG_A = 100;
const int TAG_C = 200;

int get_rows_count(int rank, int process_count, int n)
{
    int base = n / process_count;
    int remainder = n % process_count;

    if (rank < remainder) {
        return base + 1;
    }

    return base;
}

int get_start_row(int rank, int process_count, int n)
{
    int base = n / process_count;
    int remainder = n % process_count;

    if (rank < remainder) {
        return rank * (base + 1);
    }

    return remainder * (base + 1) + (rank - remainder) * base;
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

    Matrix matrix(rows, vector<int>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (!(file >> matrix[i][j])) {
                throw runtime_error("Not enough matrix elements in file: " + filename);
            }
        }
    }

    return matrix;
}

void write_matrix(const string& filename, const Matrix& matrix)
{
    ofstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Cannot open file for writing: " + filename);
    }

    int rows = static_cast<int>(matrix.size());
    int cols = rows > 0 ? static_cast<int>(matrix[0].size()) : 0;

    file << rows << ' ' << cols << '\n';

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix[i][j];

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

void write_result_file(
    const string& filename,
    const Matrix& matrix,
    int process_count,
    double elapsed_seconds,
    bool is_correct
) {
    ofstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Cannot open file for writing: " + filename);
    }

    int n = static_cast<int>(matrix.size());
    long long operations = (2LL * n - 1) * n * n;

    file << "Matrix size: " << n << " x " << n << '\n';
    file << "Processes: " << process_count << '\n';
    file << "Time: " << elapsed_seconds << " sec\n";
    file << "Time: " << elapsed_seconds * 1000.0 << " ms\n";
    file << "Number of operations: " << operations << '\n';
    file << "Correctly: " << is_correct << "\n\n";

    file << "Result matrix:\n";

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << matrix[i][j];

            if (j + 1 < n) {
                file << ' ';
            }
        }

        file << '\n';
    }
}

Matrix matrix_multiply_mpi(
    const Matrix& A,
    Matrix& B,
    int n,
    int rank,
    int process_count
) {
    if (rank != ROOT) {
        B.assign(n, vector<int>(n));
    }
    for (int i = 0; i < n; ++i) {
        MPI_Bcast(
            B[i].data(),
            n,
            MPI_INT,
            ROOT,
            MPI_COMM_WORLD
        );
    }

    int local_rows = get_rows_count(rank, process_count, n);
    int start_row = get_start_row(rank, process_count, n);

    Matrix local_A(local_rows, vector<int>(n));
    Matrix local_C(local_rows, vector<int>(n, 0));
    if (rank == ROOT) {
        for (int i = 0; i < local_rows; ++i) {
            local_A[i] = A[start_row + i];
        }

        for (int proc = 1; proc < process_count; ++proc) {
            int proc_rows = get_rows_count(proc, process_count, n);
            int proc_start = get_start_row(proc, process_count, n);

            for (int i = 0; i < proc_rows; ++i) {
                MPI_Send(
                    A[proc_start + i].data(),
                    n,
                    MPI_INT,
                    proc,
                    TAG_A,
                    MPI_COMM_WORLD
                );
            }
        }
    } else {
        for (int i = 0; i < local_rows; ++i) {
            MPI_Recv(
                local_A[i].data(),
                n,
                MPI_INT,
                ROOT,
                TAG_A,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );
        }
    }
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                local_C[i][j] += local_A[i][k] * B[k][j];
            }
        }
    }

    Matrix C;
    if (rank == ROOT) {
        C.assign(n, vector<int>(n));

        for (int i = 0; i < local_rows; ++i) {
            C[start_row + i] = local_C[i];
        }

        for (int proc = 1; proc < process_count; ++proc) {
            int proc_rows = get_rows_count(proc, process_count, n);
            int proc_start = get_start_row(proc, process_count, n);

            for (int i = 0; i < proc_rows; ++i) {
                MPI_Recv(
                    C[proc_start + i].data(),
                    n,
                    MPI_INT,
                    proc,
                    TAG_C,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );
            }
        }
    } else {
        for (int i = 0; i < local_rows; ++i) {
            MPI_Send(
                local_C[i].data(),
                n,
                MPI_INT,
                ROOT,
                TAG_C,
                MPI_COMM_WORLD
            );
        }
    }

    return C;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int process_count = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    Matrix A;
    Matrix B;
    Matrix C;

    int rows_A = 0;
    int cols_A = 0;
    int rows_B = 0;
    int cols_B = 0;
    int n = 0;
    int input_is_ok = 1;

    if (rank == ROOT) {
        try {
            A = read_matrix("matrixA.txt", rows_A, cols_A);
            B = read_matrix("matrixB.txt", rows_B, cols_B);

            if (rows_A != cols_A || rows_B != cols_B) {
                throw runtime_error("Input matrices must be square");
            }

            if (rows_A != rows_B) {
                throw runtime_error("Input matrices must have the same size");
            }

            n = rows_A;
        } catch (const exception& error) {
            input_is_ok = 0;
            cerr << "Input error: " << error.what() << endl;
        }
    }

    MPI_Bcast(&input_is_ok, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    if (!input_is_ok) {
        MPI_Finalize();
        return 1;
    }

    MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    C = matrix_multiply_mpi(A, B, n, rank, process_count);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == ROOT) {
        try {
            double elapsed_seconds = end_time - start_time;

            write_matrix("matrixC.txt", C);
            bool is_correct = run_python_check();
            write_result_file("matrixAB.txt", C, process_count, elapsed_seconds, is_correct);

            cout << "Matrix size: " << n << " x " << n << endl;
            cout << "Processes: " << process_count << endl;
            cout << "Time: " << elapsed_seconds << " sec" << endl;
            cout << "Correctly: " << is_correct << endl;
        } catch (const exception& error) {
            cerr << "Output error: " << error.what() << endl;
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
