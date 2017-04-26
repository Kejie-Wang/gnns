//
// Created by WANG Kejie on 15/03/2017.
//

#ifndef GNNS_MATRIX_H
#define GNNS_MATRIX_H

#include "define.h"
#include <cstring>
#include <iostream>

namespace gnns
{
    template<typename T>
    class Matrix
    {
    public:
        //constructor
        Matrix(T* data_=NULL, size_t rows_=0, size_t cols_=0, size_t strides_=0) :
                data(data_), rows(rows_), cols(cols_), strides(strides_)
        {
            if(strides_ == 0) strides=cols;
        }


        //overload the operator [] to return a pointer
        __host__ __device__
        inline T* operator[](size_t index) const
        {
            return reinterpret_cast<T*>(data+index*strides);
        }

        //return the data pointer
        __host__ __device__
        T* ptr() const
        {
            return reinterpret_cast<T*>(data);
        }

    public:
        size_t cols;
        size_t rows;
        size_t strides;

    private:
        T* data;
    };

}


#endif //GNNS_MATRIX_H
