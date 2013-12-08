#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "gpu_canny.cu"

long time_in_usec()
{
	struct timeval  tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec) * 1000000 + tv.tv_usec;
}

void check_cuda_error(const char *message)
{
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess) {
		fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
		exit(-1);
	}
}

int get_grid_size(int data_size, int tile_size)
{
	return data_size/tile_size + (data_size%tile_size!=0);
}

void gpu_union_find(int* gradients, int* labels, int rows, int cols)
{
	int subblock_size;
	dim3 DimThread(TILE_SIZE, TILE_SIZE);
	dim3 DimGrid(cols/TILE_SIZE, rows/TILE_SIZE);

	//calculate local solutions
	kernel1<<<DimGrid, DimThread>>>(gradients, labels, cols, rows);

	//merging scheme
	subblock_size = TILE_SIZE;
	while(subblock_size < max(rows,cols)){
		dim3 DimGrid2(get_grid_size(cols, subblock_size*4), get_grid_size(rows, subblock_size*4));
		kernel2<<<DimGrid2, dim3(4,4,16)>>>(gradients, labels, cols, rows, subblock_size);
		kernel3<<<DimGrid, DimThread>>>(labels, cols, rows);
		subblock_size *= 2;
	}

	check_cuda_error("FIND-UNION");
}

void accelerated_canny_edge_detection(uchar* src_image, uchar* dest_image, int cols, int rows, int threshold1, int threshold2)
{
	int size = cols*rows*sizeof(char);
	uchar *image_on_gpu;
	int* gradients;
	int* labels;

	cudaMalloc(&image_on_gpu, size);
	cudaMalloc(&gradients, 5*size*sizeof(int));
	cudaMalloc(&labels, 5*size*sizeof(int));
	cudaMemcpy(image_on_gpu, src_image, size, cudaMemcpyHostToDevice);
	check_cuda_error("Memory");

	dim3 DimThread(TILE_SIZE, TILE_SIZE);
	dim3 DimGrid(cols/TILE_SIZE, rows/TILE_SIZE);

	clock_t T0,T1;
	T0 = time_in_usec();

	gaussian_blur_kernel<<<DimGrid, DimThread>>>(image_on_gpu, image_on_gpu, cols, rows);
	gradient_calculation_kernel<<<DimGrid, DimThread>>>(image_on_gpu, gradients, cols, rows, true);
	remove_non_edge_pixels_kernel<<<DimGrid, DimThread>>>(gradients, cols, rows, threshold1);
	gpu_union_find(gradients, labels, cols, rows);
	mark_final_edges_kernel<<<DimGrid, DimThread>>>(image_on_gpu, gradients, labels, cols, rows, threshold2);

	T1 = time_in_usec();
	printf("Czas na GPU: %ld\n", T1-T0);

	cudaMemcpy(dest_image, image_on_gpu, size, cudaMemcpyDeviceToHost);
	cudaFree(image_on_gpu);
	cudaFree(labels);
	cudaFree(gradients);
	check_cuda_error("Error");
}

uchar* read_image_from_file(char* filename, int* cols_ptr, int* rows_ptr)
{
	FILE *f = fopen(filename, "r");
	char tmp[100];
	int i, j, x;
	fgets(tmp, 100, f);
	fgets(tmp, 100, f);
	fscanf(f, "%d %d %d", cols_ptr, rows_ptr, &x);
	int rows = *rows_ptr;
	int cols = *cols_ptr;
	uchar* image = (uchar*)malloc(cols*rows);
	for(i = 0; i < rows; i++)
		for(j = 0; j < cols; j++){
			fscanf(f, "%d", &x);
			image[GET_IDX(cols, rows, j, i)] = (unsigned int)x;
		}
	fclose(f);
	return image;
}

void write_image_to_file(char* filename, uchar* image, int cols, int rows)
{
	FILE *f = fopen(filename, "w");
	int i,j;
	fprintf(f, "P2\n%d %d\n255\n", cols, rows);
	for(i = 0; i < rows; i++)
		for(j = 0; j < cols; j++)
			fprintf(f, "%u\n", (unsigned int)image[GET_IDX(cols, rows, j, i)]);
	fclose(f);
}

int main(int argc, char** argv)
{
	int M, N;
	char filename[100], out_filename[100];
	int n;
	if(argc < 4){
		printf("Uzycie: %s <nazwa pliku>.pgm <prog1> <prog2>\n", argv[0]);
		return 0;
	}
	strcpy(filename, argv[1]);
	uchar* image = read_image_from_file(filename, &M, &N);
	uchar* image2 = (uchar*)malloc(M*N*sizeof(uchar));

	n = strlen(filename);
	accelerated_canny_edge_detection(image, image2, M, N, atoi(argv[2]), atoi(argv[3]));
	strcpy(out_filename, filename);
	strcpy(out_filename+n-4, "_edges.pgm");
	printf("%s\n", out_filename);
	write_image_to_file(out_filename, image2, M, N);

	return 0;
}
