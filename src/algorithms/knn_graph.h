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
#include "naive_construction.h"
#include "naive_construction_gpu.h"

namespace gnns
{
    template <typename Distance>
    class Knn_Graph
    {
    public:
        typedef typename Distance::ElementType ElementType;
        typedef typename Distance::DistanceType DistanceType;

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

            if(method==NAIVE) naive_construction<Distance>(points, knn_indices, knn_dists, k);
            if(method==NAIVE_GPU) naive_construction_gpu<Distance>(points, knn_indices, knn_dists, k);
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
    };
}

#endif //GNNS_KNN_GRAPH_H
