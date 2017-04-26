//
// Created by WANG Kejie on 31/03/2017.
//

#ifndef GNNS_GENERAL_H
#define GNNS_GENERAL_H

#include <stdexcept>

namespace gnns
{
    class GnnsException : public std::runtime_error
    {
    public:
        GnnsException(const char* message) : std::runtime_error(message) { }
        GnnsException(const std::string& message) : std::runtime_error(message) { }
    };

    /*
     * the build graph method
     * naive: use an brute way to build O(n*n*d)
     */
    enum BUILD_GRAPH_METHOD
    {
        NAIVE, NAIVE_GPU
    };
}

#endif //GNNS_GENERAL_H
