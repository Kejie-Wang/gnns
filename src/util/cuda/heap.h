//
// Created by WANG Kejie on 27/04/2017.
//

#pragma once

#include <stdio.h>

namespace gnns
{
namespace cuda
{

template <typename ElementType>
struct less
{
    __host__ __device__
    bool operator()(const ElementType a, const ElementType b)
    {
        return a<b;
    }
};

template <typename ElementType>
struct greater
{
    __host__ __device__
    bool operator()(ElementType a, ElementType b)
    {
        return a>b;
    }
};

template <typename ElementType, typename Compare=greater<ElementType> >
class Heap
{
public:
    __host__ __device__
    Heap(ElementType* elements=NULL, size_t capacity=0, size_t size=0, Compare c=Compare())
        : _elements(elements), _capacity(capacity), _size(size), cmp(c)
    {
        if(_elements)
            adjust();
    }

    __host__  __device__
    void replace_top(const ElementType e)
    {
        _elements[0] = e;
        percolate_down(0);
    }

    __host__ __device__
    int push(const ElementType e)
    {
        if(_size == _capacity) return -1;
        _elements[_size] = e;
        percolate_up(_size++);
        return 0;
    }

    __host__ __device__
    ElementType pop()
    {
        ElementType e = _elements[0];
        _elements[0] = _elements[--_size];
        percolate_down(0);
        return e;
    }

    __host__ __device__
    ElementType top() const
    {
        return _elements[0];
    }

    __host__ __device__
    size_t size()
    {
        return _size;
    }

    __host__ __device__
    size_t capacity()
    {
        return _capacity;
    }

    __host__ __device__
    ElementType* ptr()
    {
        return _elements;
    }

    __host__ __device__
    void sort()
    {
        while(_size != 0)
        {
            ElementType e = pop();
            _elements[_size] = e;
        }
    }

private:
    __host__ __device__
    void percolate_down(const size_t index)
    {
        ElementType e = _elements[index];

        size_t i, child;
        for(i=index; i*2+1<_size; i=child)
        {
            //find smaller child
            child = 2 * i + 1; //left child
            if(child+1 != _size && cmp(_elements[child+1], _elements[child])) child++;

            //percolate one level
            if(cmp(_elements[child], e))
                _elements[i] = _elements[child];
            else
                break;
        }
        _elements[i] = e;
    }

    __host__ __device__
    void percolate_up(const size_t index)
    {
        ElementType e = _elements[index];

        size_t parent;
        size_t i=index;
        while(i != 0)
        {
            parent = (i-1)/2;
            if(cmp(e, _elements[parent]))
                _elements[i] = _elements[parent];
            else
                break;
            i = parent;
        }
        _elements[i] = e;
    }

    __host__ __device__
    void adjust()
    {
        for(int index=(_size-2)/2;index>=0;index--)
            percolate_down(index);
    }

private:
    ElementType* _elements;
    size_t _capacity;
    size_t _size;
    Compare cmp;
};

}

}
