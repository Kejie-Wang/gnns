#include <iostream>
#include <vector>
#include "matrix.h"
#include "knn_graph.h"
#include "gnns_index.h"
#include "dist.h"
#include "io.h"
#include "define.h"
#include "params.h"
#include "evaluation.h"

using namespace std;
using namespace gnns;

const string siftsmall_base_path = "../dataset/sift/siftsmall/siftsmall_base.fvecs";
const string siftsmall_query_path = "../dataset/sift/siftsmall/siftsmall_query.fvecs";
const string siftsmall_groundtruth_path = "../dataset/sift/siftsmall/siftsmall_groundtruth.ivecs";

const string graph_index_saved_path = "../saved_graph/index.fvec";
const string graph_dist_saved_path = "../saved_graph/dist.fvec";

int main()
{
    // read the dataset and query
    Matrix<float> data = load_from_file<float>(siftsmall_base_path);

    // build the gnns index
    Gnns_Index<L2Distance<float> > gnns_index(data);
    gnns_index.build_index(graph_index_saved_path, graph_dist_saved_path);

    // do knn search
    int knn = 10;
    Matrix<float> queries = load_from_file<float>(siftsmall_query_path);
    Matrix<IndexType> indices(new IndexType[queries.rows*knn], queries.rows, knn);
    Matrix<float> dists(new float[queries.rows*knn], queries.rows, knn);
    gnns_index.knn_search(queries, indices, dists, knn, Search_Params());

    // evaluation
    Matrix<IndexType> groundtruth = load_from_file<IndexType>(siftsmall_groundtruth_path);
    float precision = compute_precision<IndexType>(indices, groundtruth);
    cout << "precision: " << precision << endl;


    delete data.ptr();
    delete queries.ptr();
    delete indices.ptr();
    delete dists.ptr();
}
