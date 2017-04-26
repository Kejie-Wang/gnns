//
// Created by WANG Kejie on 26/04/2017.
//

#pragma once

#include "matrix.h"
#include "define.h"
#include "nth_element.h"

namespace gnns
{
    /*
     * @brief use a naive construction way to build a knn nearest neighbor graph
     * @param data: the coordinate of the point in matrix type with shape [point_num, dim]
     */
    template <typename Distance>
    void naive_construction(const Matrix<typename Distance::ElementType>& points,
                            Matrix<IndexType> indices,
                            Matrix<typename Distance::DistanceType> dists,
                            const size_t k_)
    {
        typedef typename Distance::ElementType ElementType;
        typedef typename Distance::DistanceType DistanceType;

        size_t vec_num = points.rows;
        size_t vec_len = points.cols;

        Matrix<DistanceType> dist(new DistanceType[vec_num*vec_num], vec_num, vec_num);

        Distance distance;

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
            nth_index_element<ElementType, DistanceType>(dist[i], vec_num, dists[i], indices[i], k_);
        }
        delete[] dist.ptr();
    }
}
