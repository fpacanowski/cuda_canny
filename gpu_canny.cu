typedef unsigned char uchar;
const int TILE_SIZE = 16;
#define PI 3.1415
#define GET_IDX(cols, rows, x, y) (((y)*(cols))+(x))

/*** IMAGE PROCESSING KERNELS ***/
__device__ int apply_convolution(uchar*, int, int, int, int, int*, int);

__constant__ int gauss_matrix[25] =
{2,  4,  5,  4, 2,
 4,  9, 12,  9, 4,
 5, 12, 15, 12, 5,
 4,  9, 12,  9, 4,
 2,  4,  5,  4, 2};

__constant__ int horizontal_sobel_matrix[9] = 
{-1, 0, 1
 -2, 0, 2
 -1, 0, 1};

__constant__ int vertical_sobel_matrix[9] =
{-1, -2, -1,
  0,  0,  0,
  1,  2,  1};

__global__ void gaussian_blur_kernel(uchar* image, uchar* dest_image, int cols, int rows)
{
	int x = blockIdx.x*TILE_SIZE + threadIdx.x;
	int y = blockIdx.y*TILE_SIZE + threadIdx.y;
	int idx = GET_IDX(cols, rows, x, y);

	dest_image[idx] = (1.0/159)*apply_convolution(image, x, y, cols, rows, gauss_matrix, 5);
}

__global__ void remove_non_edge_pixels_kernel(int* gradients, int cols, int rows, int threshold)
{
	int x = blockIdx.x*TILE_SIZE + threadIdx.x;
	int y = blockIdx.y*TILE_SIZE + threadIdx.y;
	int idx = GET_IDX(cols, rows, x, y);

	gradients[idx] = (gradients[idx] >= threshold) ? gradients[idx] : 0;
}

__global__ void gradient_calculation_kernel(uchar* image, int* gradients, int cols, int rows, bool nonmax_suppression)
{
	int x = blockIdx.x*TILE_SIZE + threadIdx.x;
	int y = blockIdx.y*TILE_SIZE + threadIdx.y;
	int idx = GET_IDX(cols, rows, x, y);

	int Gx = apply_convolution(image, x, y, cols, rows, horizontal_sobel_matrix, 3);
	int Gy = apply_convolution(image, x, y, cols, rows, vertical_sobel_matrix, 3);
	int grad = sqrtf(Gx*Gx + Gy*Gy);
	//int grad = Gx;
	float direction = atan2f(Gy, Gx);
	__syncthreads();

	gradients[idx] = grad;
	if(!nonmax_suppression)
		return;
	__syncthreads();

	if(fabs(direction) < PI/8 || fabs(direction) > 7*PI/8){
		//east
		if(gradients[idx] < gradients[GET_IDX(cols, rows, x+1, y)] ||
		   gradients[idx] < gradients[GET_IDX(cols, rows, x-1, y)])
			grad = 0;
	}else if((PI/8 < direction && direction < 3*PI/8) || ((-5*PI/8 > direction && direction > -7*PI/8))){
		//north-east
		if(gradients[idx] < gradients[GET_IDX(cols, rows, x+1, y-1)] ||
		   gradients[idx] < gradients[GET_IDX(cols, rows, x-1, y+1)])
			grad = 0;
	}else if((3*PI/8 < direction && direction < 5*PI/8) || ((-3*PI/8 > direction && direction > -5*PI/8))){
		//north
		if(gradients[idx] < gradients[GET_IDX(cols, rows, x, y+1)] ||
		   gradients[idx] < gradients[GET_IDX(cols, rows, x, y-1)])
			grad = 0;
	}else{
		//south-east
		if(gradients[idx] < gradients[GET_IDX(cols, rows, x+1, y+1)] ||
		   gradients[idx] < gradients[GET_IDX(cols, rows, x-1, y-1)])
			grad = 0;
	}

	__syncthreads();
	gradients[idx] = grad;
}

__device__ int apply_convolution(uchar* image, int x, int y, int cols, int rows,
                                 int* matrix, int matrix_size)
{
	int idx = GET_IDX(cols, rows, x, y);
	int coef;
	int coef_sum = 0;
	int dot_product = 0;
	int dx, dy;
	int h = matrix_size/2;
	for(dx = -h; dx <= h; dx++)
		for(dy = -h; dy <= h; dy++){
			if(x+dx < 0 || x+dx >= cols || y+dy < 0)
				continue;
			idx = GET_IDX(cols, rows, x+dx, y+dy);
			coef = matrix[matrix_size*(dy+h)+(dx+h)];
			coef_sum += coef;
			dot_product += image[idx]*coef;
		}
	return dot_product;
}

/*** CCL ALGORITHM KERNELS ***/
__device__ bool check_connection(int gradient1, int gradient2);
__device__ int find_root(int* tree, int vertex);
__device__ int get_global_addr(int cols ,int rows, int block_size, int block_x, int block_y, int offset_x, int offset_y);
__device__ void union_sets(int* gradients, int* labels, int vertex1, int vertex2, int* changed);

__global__ void kernel1(int* gradients, int* labels, int cols, int rows)
{
	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	int local_x = threadIdx.x;
	int local_y = threadIdx.y;
	int local_idx = GET_IDX(TILE_SIZE, TILE_SIZE, local_x, local_y);

	int idx = GET_IDX(cols, rows, x, y);
	int grad = gradients[idx];
	__shared__ bool sChanged[1];
	__shared__ int s_labels[TILE_SIZE*TILE_SIZE];
	__shared__ int s_gradients[TILE_SIZE+2][TILE_SIZE+2];
	s_gradients[local_x+1][local_y+1] = grad;
	for(int i = 0; i < TILE_SIZE+2; i++){
		s_gradients[TILE_SIZE+1][i] = 0;
		s_gradients[0][i] = 0;
		s_gradients[i][TILE_SIZE+1] = 0;
		s_gradients[i][0] = 0;
	}
	__syncthreads();
	int label = idx;
	int new_label = label;
	int neighbour_idx;
	if(grad == 0){
		labels[idx] = 0;
		if(!(threadIdx.x == 0 && threadIdx.y == 0)) //we need thread (0,0)
			return;
	}
	labels[idx] = idx;
	label = local_idx;
	__syncthreads();
	while(1)
	{
		s_labels[local_idx] = label;
		if(threadIdx.x == 0 && threadIdx.y == 0)
			sChanged[0] = 0;
		__syncthreads();
		new_label = label;
		for(int dx = -1; dx <= 1; dx++)
			for(int dy = -1; dy <= 1; dy++){
				neighbour_idx = GET_IDX(TILE_SIZE, TILE_SIZE, local_x+dx, local_y+dy);
				if(check_connection(s_gradients[local_x+dx+1][local_y+dy+1], s_gradients[local_x+1][local_y+1]))
					new_label = min(new_label, s_labels[neighbour_idx]);
			}
		__syncthreads();
		if(new_label < label) {
			atomicMin(&s_labels[label], new_label);
			sChanged[0] = 1;
		}
		__syncthreads();
		if(sChanged[0] == 0){
			break;
		}
		label = find_root(s_labels, label);
		__syncthreads();
	}
	__syncthreads();
	int a = blockIdx.x*TILE_SIZE + s_labels[GET_IDX(TILE_SIZE, TILE_SIZE, local_x, local_y)]%16;
	int b = blockIdx.y*TILE_SIZE + s_labels[GET_IDX(TILE_SIZE, TILE_SIZE, local_x, local_y)]/16;
	labels[idx] = GET_IDX(cols, rows, a, b);
	if(grad == 0)
		labels[idx] = 0;
}

__global__ void kernel2(int* gradients, int* labels, int cols, int rows, int subblock_size)
{
	int subblock_x = threadIdx.x + blockIdx.x*blockDim.x;
	int subblock_y = threadIdx.y + blockIdx.y*blockDim.y;
	int repetitions = subblock_size/blockDim.z;
	bool top_border = (threadIdx.y == 0);
	bool bottom_border = (threadIdx.y == blockDim.y-1);
	bool left_border = (threadIdx.x == 0);
	bool right_border = (threadIdx.x == blockDim.x-1);

	__shared__ int s_changed[1];

	while(1){
		if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
			s_changed[0] = 0;

		for(int i = 0; i < repetitions && !bottom_border; i++){
			int x = threadIdx.z + i*blockDim.z;
			int y = subblock_size-1;

			int pixel_idx = get_global_addr(cols, rows, subblock_size,
			                                subblock_x, subblock_y, x, y);
			int lower_neigh_idx = get_global_addr(cols, rows, subblock_size,
			                                      subblock_x, subblock_y, x, y+1);
			int lower_right_neigh_idx = get_global_addr(cols, rows, subblock_size,
			                                            subblock_x, subblock_y, x+1, y+1);
			int lower_left_neigh_idx = get_global_addr(cols, rows, subblock_size,
			                                           subblock_x, subblock_y, x-1, y+1);

			union_sets(gradients, labels, pixel_idx, lower_neigh_idx, s_changed);
			if(!right_border)
				union_sets(gradients, labels, pixel_idx, lower_right_neigh_idx, s_changed);
			if(!left_border)
				union_sets(gradients, labels, pixel_idx, lower_left_neigh_idx, s_changed);
		}
		for(int i = 0; i < repetitions && !right_border; i++){
			int x = subblock_size-1;
			int y = threadIdx.z + i*blockDim.z;

			int pixel_idx = get_global_addr(cols, rows, subblock_size,
			                                subblock_x, subblock_y, x, y);
			int right_neigh_idx = get_global_addr(cols, rows, subblock_size,
			                                      subblock_x, subblock_y, x+1, y);
			int upper_right_neigh_idx = get_global_addr(cols, rows, subblock_size,
			                                            subblock_x, subblock_y, x+1, y-1);
			int lower_right_neigh_idx = get_global_addr(cols, rows, subblock_size,
			                                            subblock_x, subblock_y, x+1, y+1);

			union_sets(gradients, labels, pixel_idx, right_neigh_idx, s_changed);
			if(!top_border)
				union_sets(gradients, labels, pixel_idx, upper_right_neigh_idx, s_changed);
			if(!bottom_border)
				union_sets(gradients, labels, pixel_idx, lower_right_neigh_idx, s_changed);
		}
		__syncthreads();
		if(s_changed[0] == 0)
			break;
		__syncthreads();
	}
}

__global__ void kernel3(int* labels, int cols, int rows)
{
	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	int idx = GET_IDX(cols, rows, x, y);
	if(x >= cols || y >= rows)
		return;
	
	int root = find_root(labels, idx);
	__syncthreads();
	labels[idx] = root;
}

__global__ void mark_final_edges_kernel(uchar* image, int* gradients, int* labels, int cols, int rows, int threshold)
{
	int x = blockIdx.x*TILE_SIZE + threadIdx.x;
	int y = blockIdx.y*TILE_SIZE + threadIdx.y;
	int idx = GET_IDX(cols, rows, x, y);
	int super_vertex = cols*rows+1;

	if(x >= cols || y >= rows)
		return;

	if(gradients[idx] == 0){
		image[idx] = 0;
		return;
	}
	//if(threadIdx.x == 0 && threadIdx.y == 0)
		labels[super_vertex] = super_vertex;

	if(gradients[idx] >= threshold)
		labels[idx] = super_vertex;
	__syncthreads();

	int root = find_root(labels, idx);
	//int root = 7;
	image[idx] = (root == super_vertex) ? 255 : 0;
}

__device__ bool check_connection(int gradient1, int gradient2)
{
	if ((gradient1 > 0) && (gradient2 > 0))
		return true;
	return false;
}

__device__ int find_root(int* tree, int vertex)
{
	while(tree[vertex] != vertex)
		vertex = tree[vertex];
	return vertex;
}

__device__ void union_sets(int* gradients, int* labels, int vertex1, int vertex2, int* changed)
{
	if(vertex1 == -1 || vertex2 == -2)
		return;
	if(!check_connection(gradients[vertex1], gradients[vertex2]))
		return;
	int root1 = find_root(labels, vertex1);	
	int root2 = find_root(labels, vertex2);	
	
	if(root1 > root2) {
		atomicMin(labels+root1, root2);
		changed[0] = 1;
	} else if(root2 > root1) {
		atomicMin(labels+root2, root1);
		changed[0] = 1;
	}
}

__device__ int get_global_addr(int cols ,int rows, int block_size, int block_x, int block_y, int offset_x, int offset_y)
{
	int x = block_x*block_size+offset_x;
	int y = block_y*block_size+offset_y;
	if((x < 0) || (x >= cols) || (y < 0) || (y >= rows))
		return -1;
	return GET_IDX(cols, rows, x, y);
}
