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
            this->points_num = data.rows;
            this->vec_len = data.cols;
            for(int i=0;i<points_num;++i)
            {
                points.push_back(data[i]);
            }
        }

    public:
        //Default constructor
        Gnns_Index(){}

        Gnns_Index(const Matrix<ElementType>& data, Gnns_Params params = Gnns_Params())
        {
            setDataset(data);
            k = params.Graph_k;
            this->method = params.method;
        }

        /*
         *@breif build the index from the saved graph
         */
        void build_index(const std::string& graph_index_path, const std::string& graph_dist_path, bool rebuild=false)
        {
            if(rebuild)
            {
                graph.build_graph(points, vec_len, k, method);
                graph.save_graph(graph_index_path, graph_dist_path);
            }
            else
            {
                try{
                    graph.load_graph(graph_index_path, graph_dist_path);
                }catch(std::exception& e){
                    std::cout << "The saved graph is not exist and building a graph, this may cost lots of time..." << std::endl;
                    graph.build_graph(points, vec_len, k, method);
                    graph.save_graph(graph_index_path, graph_dist_path);
                }
            }
        }

        void knn_search(const Matrix<ElementType>& queries,
            Matrix<IndexType>& indices,
            Matrix<DistanceType> dists,
            size_t knn,
            const Search_Params& params)
        {
            //assert the vector in same dim
            assert(queries.cols==vec_len);
            assert(indices.rows>=queries.rows);
            assert(dists.rows>=queries.rows);
            assert(indices.cols>=knn);
            assert(dists.cols>=knn);

            if(params.E > k)
            {
                std::cout << "WARNINGS: The search expand E in param exceeds the k (param of the build knn graph)" << std::endl;
                std::cout << "WARNINGS: The procedure will use k(" << k << ")" << " as the search expand." << std::endl;
                std::cout << "WARNINGS: For using a larger search expand, you can rebuild the graph with a larger k." << std::endl;
            }
            srand((unsigned)time(0));
            //for each query
            for(int i=0;i<queries.rows;++i)
            {
                ElementType* q = queries[i];
                IndexType* in = indices[i];
                DistanceType* d = dists[i];
                find_neighbors(queries[i], indices[i], dists[i], knn, params);
            }
        }

    private:
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
                size_t v_it = rand()%points_num;
                DistanceType min_dist = -1;
                // std::cout << v_it << std::endl;
                while(true)
                {
                    std::vector<IndexType> neighbors = graph.get_neighbors(v_it, params.E);
                    std::vector<DistanceType> dist_to_query(neighbors.size());

                    for(int i=0;i<neighbors.size();++i)
                    {
                        IndexType neighbor_index = neighbors[i];
                        dist_to_query[i] = distance(points[neighbor_index], query, vec_len);
                        // std::cout << neighbor_index << "\t" << dist_to_query[i] << std::endl;
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
                        // std::cout << std::endl;
                        break;
                    }
                    // std::cout << "-->" << v_it;
                }
                if(R==0 && dist_and_index.size() < knn)
                {
                    std::cout << "R++" << std::endl;
                    R += 1;
                }
            }

            size_t k = 0;
            for(auto it = dist_and_index.begin(); it!=dist_and_index.end();++it)
            {
                dist[k] = it->first;
                index[k] = it->second;
                k++;
                if(k==knn)
                    break;
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

        //param k of knn grapn
        size_t k;

        //build graph method
        BUILD_GRAPH_METHOD method;

        //Distance
        Distance distance;

    };
}



#endif //GNNS_GNNS_INDEX_H
