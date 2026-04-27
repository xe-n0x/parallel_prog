#include "lab1_pp.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>


using namespace std;

vector<vector<int>> read_matrix(int &n, int &m, const string filename){
	ifstream file(filename);
	if (!file.is_open()) {
		cout<< "can't open the file1\n";
		exit(1);
	}
	file >> n;
	file >> m;
	vector<vector<int>> matrix(n, vector<int>(m));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			file >> matrix[i][j];
		}
	}
	file.close();
	return matrix;
}

vector<vector<int>> matrix_multiply(const vector<vector<int>> &A, const vector<vector<int>> &B) {
	int n = A.size();
	int m = B[0].size();
	int p = A[0].size();
	vector<vector<int>> C(n, vector<int>(m));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			for (int k = 0; k < p; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	return C;
}

void write_matrix(string filename, int n, int m, const vector<vector<int>> &matrix, chrono::milliseconds t, bool correct) {
	ofstream file(filename);
	if (!file.is_open()) {
		cout << "can't open the file2\n";
		exit(1);
	}
	file << n << " " << m << "\n";
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			file << matrix[i][j] << " ";
			if (j < m - 1) {
				file << " ";
			}
			file << "\n";
		}
		file << "Time: " << t.count() << " ms";
		file << "\nCorrectly: " << correct;
		file.close();
	}
}

void write_matrix(string filename, int n, int m, const vector<vector<int>> &matrix) {
	ofstream file(filename);
	if (!file.is_open()) {
		cout << "can't open the file3\n";
		exit(1);
	}
	file << n << " " << m << "\n";
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			file << matrix[i][j] << " ";
			if (j < m - 1) {
				file << " ";
			}
			file << "\n";
		}
	}
	file.close();
}

int main(){
	int nA, mA, nB, mB;
	auto start = chrono::high_resolution_clock::now();
	vector<vector<int>> A = read_matrix(nA, mA, "matrixA.txt");
	vector<vector<int>> B = read_matrix(nB, mB, "matrixB.txt");
	vector<vector<int>> C = matrix_multiply(A, B);
	auto end = chrono::high_resolution_clock::now();
	chrono::milliseconds t = chrono::duration_cast<chrono::milliseconds>(end - start);
	write_matrix("matrixC.txt", nA, mB, C);
	bool correct = (system("python3 eqw.py") == 0);
	write_matrix("matrixAB.txt", nA, mB, C, t, correct);
}