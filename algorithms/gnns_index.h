//
// Created by WANG Kejie on 19/03/2017.
//

#ifndef GNNS_GNNS_INDEX_H
#define GNNS_GNNS_INDEX_H

#include <set>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include "knn_graph.h"
#include "params.h"

namespace gnns
{
    class Gnns_Params : public Index_Params
    {
    public:
        Gnns_Params(size_t Graph_k = 1000, BUILD_GRAPH_METHOD method = NAIVE)
        {
            algorithm = "GNNS";
            this->Graph_k = Graph_k;
            this->method = method;
        }
    public:
        size_t Graph_k;
        BUILD_GRAPH_METHOD method;

    };

    template<typename Distance>
    class Gnns_Index
    {
        typedef typename Distance::ElementType ElementType;
        typedef typename Distance::DistanceType DistanceType;

    private:
        void setDataset(const Matrix<ElementType>& data)
        {
            points_num = data.rows;
            vec_len = data.cols;
            for(int i=0;i<points_num;++i)
            {
                points.push_back(data[i]);
            }
        }

    public:
        //Default constructor
        Gnns_Index(){}

        /*
         *
         */
        Gnns_Index(const Matrix<ElementType>& data, const std::string& graph_index_path, const std::string& graph_dist_path)
            : graph(graph_index_path, graph_dist_path)
        {
            setDataset(data);
        }

        Gnns_Index(const Matrix<ElementType>& data, Gnns_Params params = Gnns_Params())
            : graph(data, params.Graph_k, params.method)
        {
            setDataset(data);
        }

        void knn_search(const Matrix<ElementType>& queries,
            Matrix<IndexType>& indices,
            Matrix<DistanceType> dists,
            size_t knn,
            const Search_Params& params) const
        {
            //assert the vector in same dim
            assert(queries.cols==vec_len);
            assert(indices.rows>=queries.rows);
            assert(dists.rows>=queries.rows);
            assert(indices.cols>=knn);
            assert(dists.cols>=knn);

            //for each query
            for(int i=0;i<queries.rows;++i)
                find_neighbor(queries[i], indices[i], dists[i], knn, params);
        }

    public:
        void find_neighbors(const ElementType* query,
            IndexType* index,
            DistanceType* dist,
            size_t knn,
            const Search_Params& params)
        {
            //the random initial index numbers
            size_t R = params.R;
            std::set<std::pair<DistanceType, IndexType> > dist_and_index;

            while(R--)
            {
                //random an initial point
                srand(time(NULL));
                size_t v_it = rand()%points_num;
                DistanceType min_dist = -1;
                while(true)
                {
                    std::vector<IndexType> neighbors = graph.get_neighbors(v_it, params.E);
                    std::vector<DistanceType> dist_to_query(neighbors.size());

                    std::cout << v_it << "\t" << min_dist << std::endl;
                    std::cout << "neighbors" << std::endl;
                    for(auto i : neighbors)
                    {
                        std::cout << i << "\t";
                    }
                    std::cout << std::endl << "--------" << std::endl;
                    for(int i=0;i<neighbors.size();++i)
                    {
                        IndexType neighbor_index = neighbors[i];
                        dist_to_query[i] = distance(points[i], query, vec_len);
                        dist_and_index.insert(std::pair<DistanceType, IndexType>(dist_to_query[i], neighbor_index));
                    }
                    IndexType min_index = (std::minmax_element(dist_to_query.begin(), dist_to_query.end())).first - dist_to_query.begin();

                    if(dist_to_query[min_index] < min_dist || min_dist == -1)
                    {
                        v_it = neighbors[min_index];
                        min_dist = dist_to_query[min_index];
                    }
                    else
                    {
                        std::cout << "break" << std::endl;
                        break;
                    }
                }
                if(R==0 && dist_and_index.size() < knn)
                {
                    R += 1;
                }
            }
            size_t k = 0;
            for(auto it = dist_and_index.begin(); it!=dist_and_index.end();++it)
            {
                dist[k] = it->first;
                index[k] = it->second;
                k++;
            }
        }

    private:
        //knn graph
        Knn_Graph<Distance> graph;

        //points
        std::vector<ElementType*> points;

        //vector length
        size_t vec_len;

        //point number
        size_t points_num;

        //Distance
        Distance distance;

    };
}



#endif //GNNS_GNNS_INDEX_H
