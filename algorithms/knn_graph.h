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

namespace gnns
{
    /*
     * the build graph method
     * naive: use an brute way to build O(n*n*d)
     */
    enum BUILD_GRAPH_METHOD
    {
        NAIVE
    };

    template <typename T, typename Distance>
    class KNN_Graph
    {
    public:
        typedef typename Distance::ElementType ElementType;
        typedef typename Distance::DistanceType DistanceType;

        typedef size_t IndexType;

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
            std::vector<std::pair<DistanceType, IndexType>> v;
            for(int i=0;i<length;++i)
                v.push_back(std::pair<DistanceType, size_t>(dist[i], i));
            std::nth_element(v.begin(), v.begin()+n, v.end());
            std::sort(v.begin(), v.begin()+n);
            for(size_t i=0;i<n;++i)
            {
                elements[i] = v[i].first;
                indices[i] = v[i].second;
            }
        }


        /*
         * @brief use a naive construction way to build a knn nearest neighbor graph
         * @param data: the coordinate of the point in matrix type with shape [point_num, dim]
         */
        void naive_construction(const Matrix<T>& data)
        {
            size_t vec_num = data.rows;
            DistanceType *d = new DistanceType[vec_num*vec_num];
            Matrix<DistanceType> dist(d, vec_num, vec_num);

            for(int i=0;i<data.rows;++i)
            {
                for(int j=i+1;j<data.rows;++j)
                {
                    T* v1 = data[i];
                    T* v2 = data[j];
                    dist[i][j] = dist[j][i] = distance_(v1, v2, data.cols);
                }
            }
            for(int i=0;i<data.rows;++i)
            {
                nth_index_element(dist[i], vec_num, distances[i], indices[i], k_);
            }
        }

        void build_graph(const Matrix<T>& data, BUILD_GRAPH_METHOD method)
        {
            if(method==NAIVE)
            {
                naive_construction(data);
            }
        }


    public:

        /*Default constructor*/
        KNN_Graph(){}

        /*
         *
         * */
        KNN_Graph(Matrix<T> data, const size_t k = 1000, BUILD_GRAPH_METHOD method=NAIVE, bool rebuild = false)
        {
            k_ = k;
            indices = Matrix<IndexType>(new IndexType[data.rows*k], data.rows, k);
            distances = Matrix<DistanceType>(new DistanceType[data.rows*k], data.rows, k);
            build_graph(data, method);
        }

        ~KNN_Graph()
        {
            delete[] indices.ptr();
            delete[] distances.ptr();
        }

        /*
         * @brief load the graph from a saved graph file (index and distance)
         * @param index_file_name: the file path of the index file
         * @param dist_file_path: the file path of the distance file
         * @exception define in the io.h
         */
        void load_graph(const std::string& index_file_path, const std::string& dist_file_path)
        {
            try{
                indices = load_from_file<IndexType>(index_file_path);
                distances = load_from_file<DistanceType>(dist_file_path);
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
                save_to_file<IndexType>(indices, index_file_path);
                save_to_file<DistanceType>(distances, dist_file_path);
            }catch (std::exception& e){
                throw e;
            }
        }

    private:
        Matrix<DistanceType> distances;
        Matrix<IndexType> indices;

        size_t k_;

        Distance distance_;

    };

}

#endif //GNNS_KNN_GRAPH_H
