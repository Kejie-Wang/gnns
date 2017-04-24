//
// Created by WANG Kejie on 16/04/2017.
//

#ifndef GNNS_EVALUATION_H
#define GNNS_EVALUATION_H

#include "matrix.h"
#include <cassert>
#include <set>

namespace gnns
{
    template<typename T>
    float compute_precision(const Matrix<IndexType>& result, const Matrix<IndexType>& ground_truth)
    {
        assert(result.cols <= ground_truth.cols);
        assert(result.rows == ground_truth.rows);

        int hit = 0;
        for(int i=0;i<result.rows;++i)
        {
            for(int j=0;j<result.cols;++j)
            {
                for(int k=0;k<result.cols;++k)
                {
                    if(result[i][j] == ground_truth[i][k]) ++hit;
                }
            }
        }

        return 1.0 * hit / result.rows / result.cols;
    }
}


#endif //GNNS_EVALUATION_H
