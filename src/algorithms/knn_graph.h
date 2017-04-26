//
// Created by WANG Kejie on 15/03/2017.
//

#ifndef GNNS_KNN_GRAPH_H
#define GNNS_KNN_GRAPH_H

#include <iostream>
#include <string>
#include <exception>
#include <algorithm>
#include <vector>
#include "matrix.h"
#include "io.h"
#include "dist.h"
#include "define.h"
#include "general.h"
#include "cuda_runtime.h"
#include "device_functions.h"

namespace gnns
{
    /*
     * the build graph method
     * naive: use an brute way to build O(n*n*d)
     */
    enum BUILD_GRAPH_METHOD
    {
        NAIVE, NAIVE_GPU
    };

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
        printf("dist compute\n");
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

    template <typename Distance>
    class Knn_Graph
    {
    public:
        typedef typename Distance::ElementType ElementType;
        typedef typename Distance::DistanceType DistanceType;

    private:
        /*
         * @brief find the first n minimum elements and their index
         * @param dist: a vector which records the distance of each two points
         * @param length: the length of the dist vector
         * @param elements: the distance of first n points with minimum distances
         * @param indices: the index of the first n points with minimum distances
         * @param n: the param n
         */
        void nth_index_element(DistanceType* dist, const size_t length, DistanceType* elements, IndexType* indices, const size_t n)
        {
            //construct a vector with index
            std::vector<std::pair<DistanceType, IndexType> > v;
            for(int i=0;i<length;++i)
                v.push_back(std::pair<DistanceType, size_t>(dist[i], i));

            //since it contain the point itself and find n+1 minimum elements
            std::nth_element(v.begin(), v.begin()+n+1, v.end());
            std::sort(v.begin(), v.begin()+n+1);
            //eliminate the first element(itself)
            for(size_t i=0;i<n;++i)
            {
                elements[i] = v[i+1].first;
                indices[i] = v[i+1].second;
            }
        }

        /*
         * @brief a naive knn graph construction of gpu version
         * @params points: the points set
         * @params k_: the params of knn graph
         */
        void naive_construction_gpu(const Matrix<ElementType>& points, size_t k_)
        {
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
                std::cout << x_len << std::endl;
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

                if(i==0)
                {
                    for(int j=0;j<2;++j)
                    {
                        for(int m=0;m<10;++m)
                        {
                            std::cout << host_dist[j][m]<< " ";
                        }
                        std::cout << std::endl;
                    }
                }
                //get the first n min distances and indices
                // for(int j=0;j<x_len;++j)
                // {
                //     nth_index_element(host_dist[i], host_dist.cols, knn_dists[i+j], knn_indices[i+j], k_);
                //     for(int m=0;m<k_;++m)
                //     {
                //         std::cout << knn_indices[i+j][m] << " ";
                //     }
                //     std::cout << std::endl;
                // }

            }

            //free the device data and host data
            cudaFree(device_data);
            cudaFree(device_dist_ptr);
            delete[] host_dist_ptr;
        }

        /*
         * @brief use a naive construction way to build a knn nearest neighbor graph
         * @param data: the coordinate of the point in matrix type with shape [point_num, dim]
         */
        void naive_construction(const Matrix<ElementType>& points, const size_t k_)
        {
            size_t vec_num = points.rows;
            size_t vec_len = points.cols;

            Matrix<DistanceType> dist(new DistanceType[vec_num*vec_num], vec_num, vec_num);

            //compute the distance between each two points
            for(int i=0;i<vec_num;++i)
            {
                for(int j=i+1;j<vec_num;++j)
                {
                    ElementType* v1 = points[i];
                    ElementType* v2 = points[j];
                    dist[i][j] = dist[j][i] = distance(v1, v2, vec_len);
                }
            }

            for(int i=0;i<vec_num;++i)
            {
                nth_index_element(dist[i], vec_num, knn_dists[i], knn_indices[i], k_);
            }
            delete[] dist.ptr();
        }

    public:

        /*Default constructor*/
        Knn_Graph(){}

        ~Knn_Graph()
        {
            if(knn_indices.ptr())
                delete[] knn_indices.ptr();
            if(knn_dists.ptr())
                delete[] knn_dists.ptr();
        }

        /*
         *@brief use the data to build a k nearest graph
         *@param data: the points data in shape [point_num, vec_len]
         *@param method: the method to build the graph
         *@param k: the param k of knn graph
         */
        void build_graph(const Matrix<ElementType>& points, const size_t vec_len, const size_t k, BUILD_GRAPH_METHOD method)
        {
            size_t vec_num = points.rows;
            knn_indices = Matrix<IndexType>(new IndexType[vec_num*k], vec_num, k);
            knn_dists = Matrix<DistanceType>(new DistanceType[vec_num*k], vec_num, k);
            this->k = k;

            if(method==NAIVE) naive_construction(points, k);
            if(method==NAIVE_GPU) naive_construction_gpu(points, k);
        }

        /*
         * @brief load the graph from a saved graph file (index and distance)
         * @param index_file_name: the file path of the index file
         * @param dist_file_path: the file path of the distance file
         * @exception define in the io.h
         */
        void load_graph(const std::string& graph_index_path, const std::string& graph_dist_path)
        {
            try{
                knn_indices = load_from_file<IndexType>(graph_index_path);
                knn_dists = load_from_file<DistanceType>(graph_dist_path);
                if(knn_indices.rows != knn_dists.rows || knn_indices.cols != knn_dists.cols)
                {
                    throw GnnsException("Saved graph file error\n");
                }
                k = knn_indices.cols;
            }catch (std::exception& e){
                throw e;
            }
        }

        /*
         * @brief save the graph into the disk (index and distance)
         * @param index_file_name: the file path of the index file
         * @param dist_file_path: the file path of the distance file
         * @exception define in the io.h
         */
        void save_graph(const std::string& index_file_path, const std::string& dist_file_path)
        {
            try{
                save_to_file<IndexType>(knn_indices, index_file_path);
                save_to_file<DistanceType>(knn_dists, dist_file_path);
            }catch (std::exception& e){
                throw e;
            }
        }

        void get_neighbors(const IndexType search_index,
            std::vector<IndexType> indices,
            std::vector<DistanceType> dists,
             int graph_search_expand=-1)
        {
            if(graph_search_expand == -1 || graph_search_expand > k)
            {
                graph_search_expand = k;
            }

            indices.resize(graph_search_expand);
            dists.resize(graph_search_expand);
            for(int i=0;i<graph_search_expand;++i)
            {
                indices[i] = this->knn_indices[search_index][i];
                dists[i] = this->knn_dists[search_index][i];
            }
        }

        std::vector<IndexType> get_neighbors(const IndexType search_index, int graph_search_expand=-1)
        {
            if(graph_search_expand == -1 || graph_search_expand > k)
            {
                graph_search_expand = k;
            }

            std::vector<IndexType> neighbors;
            neighbors.resize(graph_search_expand);
            for(int i=0;i<graph_search_expand;++i)
            {
                neighbors[i] = this->knn_indices[search_index][i];
            }

            return neighbors;
        }

    private:

        //distance of the two points
        Matrix<DistanceType> knn_dists;

        //nearst neighbor index
        Matrix<IndexType> knn_indices;

        //k nearst neighbor in the graph
        size_t k;

        Distance distance;
    };
}

#endif //GNNS_KNN_GRAPH_H
