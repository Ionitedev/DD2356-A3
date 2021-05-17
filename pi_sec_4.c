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
    int provided, rank, size, sum = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

    double time = MPI_Wtime();

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

    MPI_Reduce(&count, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Estimate Pi and display the result
        pi = ((double)sum / (double)NUM_ITER) * 4.0;
        time = MPI_Wtime() - time;
        printf("The result is %f, time: %lf sec\n", pi, time);
    }

    MPI_Finalize();

    return 0;
}
