//
// Created by WANG Kejie on 26/04/2017.
//

#pragma once

#include <algorithm>
#include <vector>

namespace gnns
{
    /*
     * @brief find the first n minimum elements and their index
     * @param dist: a vector which records the distance of each two points
     * @param length: the length of the dist vector
     * @param elements: the distance of first n points with minimum distances
     * @param indices: the index of the first n points with minimum distances
     * @param n: the param n
     */
    template <typename ElementType, typename DistanceType>
    void nth_index_element(DistanceType* dist, const size_t length, DistanceType* elements, IndexType* indices, const size_t n)
    {
        //construct a vector with index
        std::vector<std::pair<DistanceType, IndexType> > v;
        v.resize(length);
        for(int i=0;i<length;++i)
            v[i] = std::pair<DistanceType, size_t>(dist[i], i);

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
}
