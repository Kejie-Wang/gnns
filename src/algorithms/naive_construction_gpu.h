//
// Created by WANG Kejie on 26/04/2017.
//

#pragma once

#include <thrust/pair.h>
#include "matrix.h"
#include "define.h"
#include "nth_element.h"
#include "heap.h"
#include <ctime>
#include <iostream>

namespace gnns
{
namespace cuda
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

    template <typename DistanceType, typename DistanceIndexType>
    __global__
    void get_nth_elements(Matrix<DistanceType> dists, Matrix<DistanceIndexType> dist_index, size_t k)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= dists.rows) return;

        DistanceType* dist = dists[index];
        size_t length = dists.cols;

        Heap<DistanceIndexType> heap(dist_index[index], k);
        for(int i=0;i<k;++i)
        {
            heap.push(DistanceIndexType(dist[i], i));
        }
        for(int i=k;i<length;++i)
        {
            if(dist[i] < heap.top().first)
                heap.replace_top(DistanceIndexType(dist[i], i));
        }
        heap.sort();
    }

    /*
     * @brief a naive knn graph construction of gpu version
     * @params points: the points set
     * @params k_: the params of knn graph
     */
    template <typename Distance>
    void naive_construction_gpu(const Matrix<typename Distance::ElementType>& points,
                                Matrix<IndexType>& knn_indices,
                                Matrix<typename Distance::DistanceType>& knn_dists,
                                size_t k)
    {
        typedef typename Distance::ElementType ElementType;
        typedef typename Distance::DistanceType DistanceType;

        typedef thrust::pair<DistanceType, IndexType> DistanceIndexType;

        //copy data from host to device
        ElementType* dev_data;
        cudaMalloc(&dev_data, sizeof(ElementType) * points.rows * points.strides);
        cudaMemcpy(dev_data, points.ptr(), sizeof(ElementType) * points.rows * points.strides, cudaMemcpyHostToDevice);
        Matrix<ElementType> dev_points(dev_data, points.rows, points.cols, points.cols);

        //device dist data allocation
        //it used for all batch
        DistanceType* dev_dist_ptr;
        cudaMalloc(&dev_dist_ptr, sizeof(DistanceType) * BATCH_SIZE_X * points.rows);

        DistanceIndexType* dev_dist_index_ptr;
        cudaMalloc(&dev_dist_index_ptr, sizeof(DistanceIndexType) * BATCH_SIZE_X * k);

        //host dist and inde data allocation
        DistanceIndexType* host_dist_index_ptr = new DistanceIndexType[BATCH_SIZE_X*k];

        int time_dist=0, time_sort=0;

        //compute [BATCH_SIZE_X, BATCH_SIZE_Y] distance every kernel
        //for the outer iteration, it nerges all BATCH_SIZE_X distances and get the first k min distances and indices
        for(int i=0;i<points.rows;i+=BATCH_SIZE_X)
        {
            clock_t start;
            start = clock();

            size_t x_len = points.rows-i>BATCH_SIZE_X ? BATCH_SIZE_X : points.rows-i;
            Matrix<DistanceType> dev_dists(dev_dist_ptr, x_len, points.rows, points.rows);
            Matrix<DistanceIndexType> dev_dist_index(dev_dist_index_ptr, x_len, k, k);
            Matrix<DistanceIndexType> host_dist_index(host_dist_index_ptr, x_len, k, k);

            //the subset of points in x axes
            Matrix<ElementType> points_x(dev_points[i], x_len, dev_points.cols, dev_points.strides);

            for(int j=0;j<points.rows;j+=BATCH_SIZE_Y)
            {
                size_t y_len = points.rows-j>BATCH_SIZE_Y ? BATCH_SIZE_Y : points.rows-j;

                Matrix<ElementType> points_y(dev_points[j], y_len, dev_points.cols, dev_points.strides);
                Matrix<DistanceType> dist_xy(dev_dists.ptr() + j, x_len, y_len, dev_dists.strides);

                dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
                dim3 block_num((x_len+block_size.x-1)/block_size.x, (y_len+block_size.y-1)/block_size.y);
                dist_compute<ElementType, DistanceType><<<block_num, block_size>>>(points_x, points_y, dist_xy);
            }

            time_dist += clock() - start;

            start = clock();

            dim3 block_size(BLOCK_SIZE);
            dim3 block_num((x_len+block_size.x-1) / block_size.x);
            get_nth_elements<DistanceType, DistanceIndexType><<<block_num, block_size>>>(dev_dists, dev_dist_index, k);
            cudaMemcpy(host_dist_index.ptr(), dev_dist_index.ptr(), sizeof(DistanceIndexType)*x_len*dev_dist_index.strides, cudaMemcpyDeviceToHost);

            for(int j=0;j<x_len;++j)
            {
                for(int m=0;m<k;++m)
                {
                    knn_dists[i+j][m] = host_dist_index[j][m].first;
                    knn_indices[i+j][m] = host_dist_index[j][m].second;
                }
            }
            // cudaMemcpy()
            time_sort += clock() - start;
        }

        std::cout << "time for computing the distance: " << time_dist / 1000000.0 << "s" << std::endl;
        std::cout << "time for getting the first n elememet: " << time_sort / 1000000.0 << "s" << std::endl;


        //free the device data and host data
        cudaFree(dev_data);
        cudaFree(dev_dist_ptr);
        cudaFree(dev_dist_index_ptr);
    }
}
}
