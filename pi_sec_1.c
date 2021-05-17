#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define SEED     921
#define NUM_ITER 1000000000

int main(int argc, char* argv[]) {
    double time = MPI_Wtime();

    int provided, rank, size;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

    int count = 0;
    double x, y, z, pi;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    srand(SEED * (rank + 1)); // Important: Multiply SEED by "rank" when you introduce MPI!
    
    // Calculate PI following a Monte Carlo method
    for (int iter = rank; iter < NUM_ITER; iter += size)
    {
        // Generate random (X,Y) points
        x = (double)random() / (double)RAND_MAX;
        y = (double)random() / (double)RAND_MAX;
        z = sqrt((x*x) + (y*y));
        
        // Check if point is in unit circle
        if (z <= 1.0)
        {
            count++;
        }
    }

    if (rank == 0) {
        int thread_count;
        for (int i = 1; i < size; i++) {
            MPI_Recv(&thread_count, 1, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            count += thread_count;
        }

        // Estimate Pi and display the result
        pi = ((double)count / (double)NUM_ITER) * 4.0;
        time = MPI_Wtime() - time;
        printf("The result is %f, time: %lf sec\n", pi, time);
    }
    else {
        MPI_Send(&count, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
