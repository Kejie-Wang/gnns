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
    Matrix<float> query = load_from_file<float>(siftsmall_query_path);

    Gnns_Index<L2Distance<float> > gnns_index(data, graph_index_saved_path, graph_dist_saved_path);

    int knn = 1;
    Matrix<IndexType> indices(new IndexType[knn], 1, knn);
    Matrix<float> dists(new float[knn], 1, knn);

    gnns_index.find_neighbors(query[0], indices[0], dists[0], knn, Search_Params());

    delete data.ptr();
    delete query.ptr();
    delete indices.ptr();
    delete dists.ptr();
}
