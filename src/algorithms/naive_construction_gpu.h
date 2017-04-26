//
// Created by WANG Kejie on 26/04/2017.
//

#include "matrix.h"
#include "define.h"
#include "nth_element.h"

namespace gnns
{
    /*
     * @brief kernel function used to compute the two set distances
     * @params points_1: points set 1 in shape [point_num_1, vec_len]
     * @params points_2: points set 2 in shape [point_num_2, vec_len]
     * @params dists: the distance in shape [point_num_1, point_num_2]
     *                 in which dists[i][j] is the distance of points_1[i] and points_2[j]
     */
     template <typename ElementType, typename DistanceType>
    __global__
    void dist_compute(Matrix<ElementType> points_1, Matrix<ElementType> points_2, Matrix<DistanceType> dists)
    {
        size_t r = threadIdx.x;
        size_t c = threadIdx.y;
        size_t vec_len = points_1.cols;

        DistanceType value = 0;
        for(int i=0;i<vec_len;i+=BLOCK_SIZE)
        {
            //fetch a block of points set from the global memory to the shared memory
            __shared__ ElementType p1[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ ElementType p2[BLOCK_SIZE][BLOCK_SIZE];
            if(blockIdx.x * blockDim.x + r < points_1.rows && c + i < points_1.cols)
                p1[r][c] = points_1[blockIdx.x * blockDim.x + r][c+i];
            else
                p1[r][c] = 0;

            if(blockIdx.y * blockDim.y + r < points_2.rows && c + i < points_2.cols)
                p2[r][c] = points_2[blockIdx.y * blockDim.y + r][c + i];
            else
                p2[r][c] = 0;

            //sync all threads to finish the data fetching into shared memory
            __syncthreads();

            for(int j=0;j<BLOCK_SIZE;++j)
            {
                DistanceType diff = p1[r][j] - p2[c][j];
                value += diff * diff;
            }

            //sync all threads in this block to finish the distance computation
            __syncthreads();
        }

        //set the distance element
        if(blockIdx.x * blockDim.x + r < points_1.rows && blockIdx.y * blockDim.y + c < points_2.rows)
            dists[blockIdx.x * blockDim.x + r][blockIdx.y * blockDim.y + c] = value;
    }


    /*
     * @brief a naive knn graph construction of gpu version
     * @params points: the points set
     * @params k_: the params of knn graph
     */
    template<typename Distance>
    void naive_construction_gpu(const Matrix<typename Distance::ElementType>& points,
                                Matrix<IndexType>& indices,
                                Matrix<typename Distance::DistanceType>& dists,
                                size_t k_)
    {
        typedef typename Distance::ElementType ElementType;
        typedef typename Distance::DistanceType DistanceType;
        
        //copy data from host to device
        ElementType* device_data;
        cudaMalloc(&device_data, sizeof(ElementType) * points.rows * points.strides);
        cudaMemcpy(device_data, points.ptr(), sizeof(ElementType) * points.rows * points.strides, cudaMemcpyHostToDevice);
        Matrix<ElementType> device_points(device_data, points.rows, points.cols, points.cols);

        //device dist data allocation
        //it used for all batch
        DistanceType* device_dist_ptr;
        cudaMalloc(&device_dist_ptr, sizeof(DistanceType) * BATCH_SIZE_X * points.rows);

        //host dist data allocation
        DistanceType* host_dist_ptr;
        host_dist_ptr = new DistanceType[BATCH_SIZE_X*points.rows];

        //compute [BATCH_SIZE_X, BATCH_SIZE_Y] distance every kernel
        //for the outer iteration, it nerges all BATCH_SIZE_X distances and get the first k min distances and indices
        for(int i=0;i<points.rows;i+=BATCH_SIZE_X)
        {
            size_t x_len = points.rows-i>BATCH_SIZE_X ? BATCH_SIZE_X : points.rows-i;
            Matrix<DistanceType> device_dists(device_dist_ptr, x_len, points.rows, points.rows);
            for(int j=0;j<points.rows;j+=BATCH_SIZE_Y)
            {
                size_t y_len = points.rows-j>BATCH_SIZE_Y ? BATCH_SIZE_Y : points.rows-j;

                Matrix<ElementType> device_points_sub_1(device_points.ptr()+i*device_points.strides, x_len, device_points.cols, device_points.strides);
                Matrix<ElementType> device_points_sub_2(device_points.ptr()+j*device_points.strides, y_len, device_points.cols, device_points.strides);
                Matrix<ElementType> device_dists_sub(device_dists.ptr() + j, x_len, y_len, device_dists.strides);

                dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
                dim3 block_num((x_len+BLOCK_SIZE-1)/BLOCK_SIZE, (y_len+BLOCK_SIZE-1)/BLOCK_SIZE);
                dist_compute<ElementType, DistanceType><<<block_num, block_size>>>(device_points_sub_1, device_points_sub_2, device_dists_sub);
            }

            //copy the distance from device to host
            Matrix<float> host_dist(host_dist_ptr, x_len, points.rows, points.rows);
            cudaMemcpy(host_dist.ptr(), device_dists.ptr(), sizeof(float)*x_len*device_dists.strides, cudaMemcpyDeviceToHost);

            for(int j=0;j<x_len;++j)
            {
                nth_index_element<ElementType, DistanceType>(host_dist[j], points.rows, dists[i+j], indices[i+j], k_);
            }
        }

        //free the device data and host data
        cudaFree(device_data);
        cudaFree(device_dist_ptr);
        delete[] host_dist_ptr;
    }

}
