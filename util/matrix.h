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
        Matrix(T* data_=nullptr, size_t rows_=0, size_t cols_=0) :
                data(data_), rows(rows_), cols(cols_)
        {
        }

        Matrix(const Matrix<T>& m)
        {
            cols = m.cols;
            rows = m.rows;
            data = new T[cols*rows];
            memcpy(data, m.data, cols*rows*sizeof(T));
        }

        //overload the operator [] to return a pointer
        inline T* operator[](size_t index) const
        {
            return reinterpret_cast<T*>(data+index*cols);
        }

        //return the data pointer
        T* ptr() const
        {
            return reinterpret_cast<T*>(data);
        }

    public:
        size_t cols;
        size_t rows;

    private:
        T* data;
    };

}


#endif //GNNS_MATRIX_H
