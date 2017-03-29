//
// Created by WANG Kejie on 19/03/2017.
//

#ifndef GNNS_PARAMS_H
#define GNNS_PARAMS_H

#include <string>
#include "define.h"

struct Search_Params
{

    Search_Params(size_t R_=1, size_t E_=100)
    {
        R = R_;
        E = E_;
    }
    // random search num
    size_t R;

    // search expand
    size_t E;
};

class Index_Params
{
public:
    std::string algorithm;
};


#endif //GNNS_DEFINE_H
