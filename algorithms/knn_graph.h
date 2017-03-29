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
            std::vector<std::pair<DistanceType, IndexType>> v;
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
         * @brief use a naive construction way to build a knn nearest neighbor graph
         * @param data: the coordinate of the point in matrix type with shape [point_num, dim]
         */
        void naive_construction(const Matrix<ElementType>& data)
        {
            size_t vec_num = data.rows;
            DistanceType *d = new DistanceType[vec_num*vec_num];
            Matrix<DistanceType> dist(d, vec_num, vec_num);

            //compute the distance between each two points
            for(int i=0;i<data.rows;++i)
            {
                for(int j=i+1;j<data.rows;++j)
                {
                    ElementType* v1 = data[i];
                    ElementType* v2 = data[j];
                    dist[i][j] = dist[j][i] = distance(v1, v2, data.cols);
                }
            }
            for(int i=0;i<data.rows;++i)
            {
                nth_index_element(dist[i], vec_num, distances[i], indices[i], k);
            }
        }

        void build_graph(const Matrix<ElementType>& data, BUILD_GRAPH_METHOD method)
        {
            if(method==NAIVE)
            {
                naive_construction(data);
            }
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
                indices = load_from_file<IndexType>(graph_index_path);
                distances = load_from_file<DistanceType>(graph_dist_path);
            }catch (std::exception& e){
                throw e;
            }
        }

    public:

        /*Default constructor*/
        Knn_Graph(){}

        /*
         *
         * */
        Knn_Graph(const Matrix<ElementType>& data, const size_t k = 1000, BUILD_GRAPH_METHOD method=NAIVE)
        {
            this->k = k;
            indices = Matrix<IndexType>(new IndexType[data.rows*k], data.rows, k);
            distances = Matrix<DistanceType>(new DistanceType[data.rows*k], data.rows, k);
            build_graph(data, method);
        }

        /*
         *
         * */
        Knn_Graph(const std::string& graph_index_path, const std::string& graph_dist_path)
        {
            load_graph(graph_index_path, graph_dist_path);
            k = indices.cols;
        }


        ~Knn_Graph()
        {
            delete[] indices.ptr();
            delete[] distances.ptr();
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

        void get_neighbors(const IndexType search_index,
            std::vector<IndexType> indices,
            std::vector<DistanceType> dists,
             int graph_search_expand=-1)
        {
            if(graph_search_expand == -1)
            {
                graph_search_expand = k;
            }

            indices.resize(graph_search_expand);
            dists.resize(graph_search_expand);
            for(int i=0;i<graph_search_expand;++i)
            {
                indices[i] = this->indices[search_index][i];
                dists[i] = this->distances[search_index][i];
            }
        }

        std::vector<IndexType> get_neighbors(const IndexType search_index, int graph_search_expand=-1)
        {
            if(graph_search_expand == -1)
            {
                graph_search_expand = k;
            }

            std::vector<IndexType> neighbors;
            neighbors.resize(graph_search_expand);
            for(int i=0;i<graph_search_expand;++i)
            {
                neighbors[i] = this->indices[search_index][i];
            }

            return neighbors;
        }

    private:

        //distance of the two points
        Matrix<DistanceType> distances;

        //nearst neighbor index
        Matrix<IndexType> indices;

        //k nearst neighbor in the graph
        size_t k;

        Distance distance;
    };

}

#endif //GNNS_KNN_GRAPH_H
