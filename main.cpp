#include <iostream>
#include <vector>
#include "matrix.h"
#include "knn_graph.h"
#include "gnns_index.h"
#include "dist.h"
#include "io.h"
#include "define.h"
#include "params.h"

using namespace std;
using namespace gnns;

const string siftsmall_base_path = "../dataset/sift/siftsmall/siftsmall_base.fvecs";
const string siftsmall_query_path = "../dataset/sift/siftsmall/siftsmall_query.fvecs";

const string graph_index_saved_path = "../saved_graph/index.fvec";
const string graph_dist_saved_path = "../saved_graph/dist.fvec";

int main()
{
    Matrix<float> data = load_from_file<float>(siftsmall_base_path);
    Matrix<float> queries = load_from_file<float>(siftsmall_query_path);

    Gnns_Index<L2Distance<float> > gnns_index(data);
    gnns_index.build_index(graph_index_saved_path, graph_dist_saved_path);


    int knn = 10;
    Matrix<IndexType> indices(new IndexType[queries.rows*knn], queries.rows, knn);
    Matrix<float> dists(new float[queries.rows*knn], queries.rows, knn);

    for(int i=0;i<queries.rows;++i)
    {
        for(int j=0;j<knn;++j)
        {
            indices[i][j];
            dists[i][j];
        }
    }

    gnns_index.knn_search(queries, indices, dists, knn, Search_Params());

    delete data.ptr();
    delete queries.ptr();
    delete indices.ptr();
    delete dists.ptr();
}
