//
// Created by WANG Kejie on 15/03/2017.
//

#ifndef GNNS_DIST_H
#define GNNS_DIST_H

// refer the flann library
template<typename T>
struct Accumulator { typedef T Type; };
template<>
struct Accumulator<unsigned char>  { typedef float Type; };
template<>
struct Accumulator<unsigned short> { typedef float Type; };
template<>
struct Accumulator<unsigned int> { typedef float Type; };
template<>
struct Accumulator<char>   { typedef float Type; };
template<>
struct Accumulator<short>  { typedef float Type; };
template<>
struct Accumulator<int> { typedef float Type; };

//Base class
template <typename T>
class Distance
{
public:
    typedef typename Accumulator<T>::Type DistanceType;

public:
    template <typename Iterator1, typename Iterator2>
    DistanceType operator()(Iterator1 a, Iterator2 b, unsigned int length){}
};

//Euclidean distance
template <typename T>
class L2Distance : public Distance<T>
{
public:
    typedef T ElementType;
    typedef typename Accumulator<T>::Type DistanceType;

    template <typename Iterator1, typename Iterator2>
    DistanceType operator()(Iterator1 a, Iterator2 b, unsigned int length)
    {
        DistanceType result = DistanceType();
        for(int i=0;i<length;++i)
        {
            DistanceType diff = *a++ - *b++;
            result += diff * diff;
        }
        return result;
    }

};

#endif //GNNS_DIST_H
