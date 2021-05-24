#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
	/* -------------------------------------------------------------------------------------------
		MPI Initialization 
	--------------------------------------------------------------------------------------------*/
	MPI_Init(&argc, &argv);

	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Status stat;

	if(size != 2){
		if(rank == 0){
			printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);
		}
		MPI_Finalize();
		exit(0);
	}

	/* -------------------------------------------------------------------------------------------
		Loop from 8 B to 1 GB
	--------------------------------------------------------------------------------------------*/

	for(int i=0; i<=27; i++){

		long int N = 1 << i;
	
   	 	// Allocate memory for A on CPU
		double *local_mem = (double*)malloc(N*sizeof(double));
        double *shared_mem = (double*)malloc(N*sizeof(double));
		// Initialize all elements of A to 0.0
		for(int i=0; i<N; i++){
			shared_mem[i] = 0.0;
            local_mem[i] = rank+i;
		}
	
		int tag1 = 10;
		int tag2 = 20;
	
		int loop_count = 50;

		// Warm-up loop
		for(int i=1; i<=5; i++){
			if(rank == 0){
				MPI_Send(shared_mem, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
				MPI_Recv(shared_mem, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
			}
			else if(rank == 1){
				MPI_Recv(shared_mem, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
				MPI_Send(shared_mem, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
			}
		}
        // Create window shared

        MPI_Win win;
        MPI_Win_create(shared_mem, N*8, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

		// Time ping-pong for loop_count iterations of data transfer size 8*N bytes
		double start_time, stop_time, elapsed_time;
		start_time = MPI_Wtime();
	
		for(int i=1; i<=loop_count; i++){
			if(rank == 0){
                MPI_Win_fence(0, win);
                MPI_Get(local_mem, N, MPI_DOUBLE, 1, 0, N, MPI_DOUBLE, win);
                MPI_Win_fence(0, win);
			}
			else if(rank == 1){
                MPI_Win_fence(0, win);
                MPI_Get(local_mem, N, MPI_DOUBLE, 0, 0, N, MPI_DOUBLE, win);
                MPI_Win_fence(0, win);
			}
		}

		stop_time = MPI_Wtime();
		elapsed_time = stop_time - start_time;

		long int num_B = 8*N;
		long int B_in_GB = 1 << 30;
		double num_GB = (double)num_B / (double)B_in_GB;
		double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

		if(rank == 0) printf("%10li\t%15.9f\n", num_B, avg_time_per_transfer);

		free(shared_mem);
        free(local_mem);
	}

	MPI_Finalize();

	return 0;
}
