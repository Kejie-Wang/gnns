//
// Created by WANG Kejie on 15/03/2017.
//

#ifndef GNNS_IO_H
#define GNNS_IO_H

#include "matrix.h"
#include <string>
#include <exception>

namespace gnns
{
    template <typename T>
    void save_to_file(const Matrix<T>& m, const std::string& file_name)
    {
        FILE *fp = fopen(file_name.c_str(), "wb");

        if(fp == NULL)
        {
            throw std::invalid_argument("const std::string file_name");
        }

        for(int i=0;i<m.rows;++i)
        {
            //write the dim
            fwrite(&m.cols, 4, 1, fp);
            //write the row data
            fwrite(m[i], sizeof(T)*m.cols, 1, fp);
        }

        fclose(fp);
    }

    template <typename T>
    Matrix<T> load_from_file(const std::string& file_name)
    {
        FILE *fp = fopen(file_name.c_str(), "rb");

        if (fp == NULL)
        {
            throw std::invalid_argument("const std::string file_name");
        }

        //read the vector dimension
        int dim;
        fread(&dim, 4, 1, fp);

        int col_size = 1 * 4 + sizeof(T) * dim;
        fseek(fp, 0, SEEK_END);
        int col = int(ftell(fp)) / col_size;

        //allocate memory for the vectors
        try {
            T* vecs = new T[dim * col];
            fseek(fp, 0, SEEK_SET);
            for (int i = 0; i < col; ++i)
            {
                int dim_;
                fread(&dim_, 4, 1, fp);
                if (dim_ != dim)
                {
                    throw std::length_error("The dimensions of all rows are NOT same!\n");
                }
                fread(vecs + i * dim, sizeof(T), dim, fp);
            }
            fclose(fp);

            return Matrix<T>(vecs, col, dim);
        } catch (const std::bad_alloc &ba) {
            throw ba;
        }
    }
}

#endif //GNNS_IO_H
