#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define N 7200
#define true 1
#define false 0

struct Matrix {
    double **data;
    int valid;
};

void buffer_read(double *buffer, double **a, int n) {
    for (int i = 0; i < n * n; i++)
        a[i / n][i % n] = buffer[i];
}

void buffer_write(double *buffer, double **a, int n) {
    for (int i = 0; i < n * n; i++)
        buffer[i] = a[i / n][i % n];
}

void map_init(struct Matrix ***a, int n) {
    *a = malloc(n * sizeof(struct Matrix*));
    for (int i = 0; i < n; i++)
        (*a)[i] = malloc(n * sizeof(struct Matrix));
}

void alloc_mat(double ***a, int n) {
    *a = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++)
        (*a)[i] = malloc(n * sizeof(double));
}

void free_mat(double **a, int n) {
    for (int i = 0; i < n; i++)
        free(a[i]);
    free(a);
}

void randomize(double **a, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            a[i][j] = (double)rand() / RAND_MAX;
}

void zero(double **a, int n) {
    for (int i = 0; i < n; i++)
        memset(a[i], 0, n * sizeof(double));
}

void add(double **a, double **b, double **c, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            c[i][j] = a[i][j] + b[i][j];
}

void add_self(double **a, double **b, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            a[i][j] += b[i][j];
}

void local_mul(double **a, double **b, double **c, int n) {
    // Loop nest optimized algorithm
    zero(c, n);
    for (int i = 0 ; i < n ; i++)
        for (int k = 0 ; k < n ; k++)
            for (int j = 0 ; j < n ; j++)
                c[i][j] += a[i][k] * b[k][j];
}

// for test
void sub_matrix(double **a, double **b, int start_col, int start_row, int sub_len) {
    for (int i = 0; i < sub_len; i++)
        for (int j = 0; j < sub_len; j++)
            b[i][j] = a[i + start_col][j + start_row];
}

double sum(double **a, int n) {
    double s = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            s += a[i][j];
    return s;
}

int main(int argc, char **argv) {
    time_t seconds;
    time(&seconds);
    srand(seconds);

    // double global_sum = 0;
    int rank, rank_row, rank_col, size, provided;
    MPI_Comm comm_cart, comm_row, comm_col;

    double **A, **B, **C;
    alloc_mat(&A, N);
    alloc_mat(&B, N);
    // alloc_mat(&C, N);
    randomize(A, N);
    randomize(B, N);

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t = MPI_Wtime();

    // if (rank == 0) {
    //     local_mul(A, B, C, N);
    //     printf("serial sum: %lf\n", sum(C, N));
    // }

    int q = sqrt(size + 1e-3);
    if (N % q != 0) {
        MPI_Finalize();
        if (rank == 0) printf("invalid size\n");
        return 0;
    }

    int dim_size[2] = {q, q};
    int period[2] = {true, true};
    int remain_row[2] = {true, false}, remain_col[2] = {false, true};

    MPI_Cart_create(MPI_COMM_WORLD, 2, dim_size, period, true, &comm_cart);
    MPI_Cart_sub(comm_cart, remain_row, &comm_row);
    MPI_Cart_sub(comm_cart, remain_col, &comm_col);

    MPI_Comm_rank(comm_cart, &rank);
    MPI_Comm_rank(comm_row, &rank_row);
    MPI_Comm_rank(comm_col, &rank_col);

    // On process (i, j):
    // Cij = 0
    // for s = 0 to q − 1
    //     k = (i + s) mod q
    //     broadcast Ai,k across process row i
    //     Ci,j = Ci,j + Ai,kBk,j
    //     if s != q − 1
    //         send Bk,j to ((i − 1) mod q, j)
    //         receive Bk+1,j from ((i + 1) mod q, j)

    int i = rank_row, j = rank_col;
    int mat_size = N / q;
    double *buffer_send = malloc(mat_size * mat_size * sizeof(double));
    double *buffer_recv = malloc(mat_size * mat_size * sizeof(double));
    double *buffer_bcast = malloc(mat_size * mat_size * sizeof(double));
    // double local_sum = 0;

    struct Matrix **map_A, **map_B;
    map_init(&map_A, q);
    map_A[i][j].valid = 5;
    alloc_mat(&map_A[i][j].data, mat_size);
    // sub_matrix(A, map_A[i][j].data, rank_row * mat_size, rank_col * mat_size, mat_size);
    randomize(map_A[i][j].data, mat_size);

    map_init(&map_B, q);
    map_B[i][j].valid = 5;
    alloc_mat(&map_B[i][j].data, mat_size);
    // sub_matrix(B, map_B[i][j].data, rank_row * mat_size, rank_col * mat_size, mat_size);
    randomize(map_B[i][j].data, mat_size);

    double **local_C;
    alloc_mat(&local_C, mat_size);
    zero(local_C, mat_size);

    for (int s = 0; s < q; s++) {
        int k = (i + s) % q;
        // broadcast
        if (j == k) {
            buffer_write(buffer_bcast, map_A[i][j].data, mat_size);
            MPI_Bcast(buffer_bcast, mat_size * mat_size, MPI_DOUBLE, k, comm_col);            
            // printf("step %d broadcast send ok!\n", s);
        }
        else {
            MPI_Bcast(buffer_bcast, mat_size * mat_size, MPI_DOUBLE, k, comm_col);
            alloc_mat(&map_A[i][k].data, mat_size);
            map_A[i][k].valid = 5;
            buffer_read(buffer_bcast, map_A[i][k].data, mat_size);
            // printf("step %d broadcast recv ok!\n", s);
        }

        double **temp;
        alloc_mat(&temp, mat_size);
        local_mul(map_A[i][k].data, map_B[k][j].data, temp, mat_size);
        add_self(local_C, temp, mat_size);
        free_mat(temp, mat_size);

        // printf("step %d mul and add ok!\n", s);

        // send Bk,j to ((i − 1) mod q, j)
        // receive Bk+1,j from ((i + 1) mod q, j)
        if (s != q - 1) {
            int dest, source;
            dest = (i - 1 + q) % q;
            source = (i + 1) % q;
            
            buffer_write(buffer_send, map_B[k][j].data, mat_size);            
            MPI_Sendrecv(buffer_send, mat_size * mat_size, MPI_DOUBLE, dest, dest, buffer_recv, mat_size * mat_size, MPI_DOUBLE, source, i, comm_row, MPI_STATUS_IGNORE);
            
            alloc_mat(&map_B[(k + 1) % q][j].data, mat_size);
            map_B[(k + 1) % q][j].valid = 5;
            buffer_read(buffer_recv, map_B[(k + 1) % q][j].data, mat_size);
            // printf("step %d sendrecv ok\n", s);
        }
    }

    free(buffer_send);
    free(buffer_recv);
    free(buffer_bcast);

    // local_sum = sum(local_C, mat_size);
    // MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm_cart);

    for (int x = 0; x < q; x++)
        for (int y = 0; y < q; y++) {
            if (map_A[x][y].valid == 5) free_mat(map_A[x][y].data, mat_size);
            if (map_B[x][y].valid == 5) free_mat(map_B[x][y].data, mat_size);
        }

    if (rank != 0) MPI_Finalize();

    if (rank == 0) {
        // printf("serial sum: %lf\n", global_sum);
        printf("time: %lf s\n", MPI_Wtime() - t);
        MPI_Finalize();
    }

    free_mat(A, N);
    free_mat(B, N);
    // free_mat(C, N);

    return 0;
}