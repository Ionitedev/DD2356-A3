#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

    int i, n;
    MPI_Comm_rank(MPI_COMM_WORLD, &i);
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    printf("Hello World from rank %d from %d processes!\n", i, n);

    MPI_Finalize();
    
    return 0;
}