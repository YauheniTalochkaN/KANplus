#pragma once

#include <vector>
#include <iostream>
#include <stdexcept>

namespace KAN
{
    template <typename T>
    class Tensor 
    {
    public:
        Tensor()
        {
            total_size = 0;
        }
    
        Tensor(const std::vector<size_t> & shape) : shape(shape), total_size(1) 
        {
            compute_strides();
            
            for (size_t dim : shape) 
            {
                total_size *= dim;
            }
    
            data.resize(total_size);
        }
    
        Tensor(const std::vector<size_t> & shape, const T & value) : shape(shape), total_size(1) 
        {
            compute_strides();
            
            for (size_t dim : shape) 
            {
                total_size *= dim;
            }
    
            data.resize(total_size, value);
        }
    
        Tensor(const std::vector<size_t> & shape, const std::vector<T> & input_data) : shape(shape), total_size(1) 
        {
            compute_strides();
            
            for (size_t dim : shape) 
            {
                total_size *= dim;
            }
    
            if (input_data.size() != total_size) 
            {
                throw std::invalid_argument("Tensor::Tensor: Wrong number of elements in the input data vector.");
            }
    
            data = input_data;
        }

        Tensor(const Tensor & other) : data(other.data), shape(other.shape), total_size(other.total_size) 
        {
            compute_strides();
        }
    
        ~Tensor()
        {
            clear();
        }
    
        void clear()
        {
            data.clear();
            shape.clear();
            strides.clear();
            total_size = 0;
        }
    
        void set_value(const T & value)
        {
            std::fill(data.begin(), data.end(), value);
        }
    
        T & operator () (const std::vector<size_t> & indices) 
        {
            return data[compute_index(indices)];
        }
    
        const T & operator () (const std::vector<size_t> & indices) const 
        {
            return data[compute_index(indices)];
        }

        std::vector<T> get_slice_1d(const size_t & index) const
        {
            if(this->get_shape().size() != 2)
            {
                throw std::runtime_error("Tensor::operator(const size_t &) const: The tensor shape size should be 2.");
            }
            
            std::vector<T> out;
            out.resize(this->get_shape().at(1), 0);

            for(size_t i = 0; i < out.size(); ++i)
            {
                out[i] = data[compute_index({index, i})];
            }
            
            return out;
        }

        Tensor & operator = (const Tensor & other) 
        {
            if (this == &other) 
            {
                return *this;
            }
            
            clear();
            
            shape = other.shape;
            compute_strides();
            total_size = other.total_size;
            data = other.data;

            return *this;
        }
    
        const std::vector<size_t> & get_shape() const 
        {
            return shape;
        }
    
        void print() const 
        {
            std::cout << "Tensor(shape=[";
            for (size_t i = 0; i < shape.size(); ++i) 
            {
                std::cout << shape[i] << (i + 1 == shape.size() ? "" : ", ");
            }
    
            std::cout << "], data=[";
    
            for (size_t i = 0; i < total_size; ++i) 
            {
                std::cout << data[i] << (i + 1 == total_size ? "" : ", ");
            }
    
            std::cout << "])" << std::endl;
        }
    
    private:
        std::vector<T> data;
        std::vector<size_t> shape;
        std::vector<size_t> strides;
        size_t total_size;

        void compute_strides() 
        {
            strides.resize(shape.size());

            size_t stride = 1;
            for (size_t i = shape.size(); i-- > 0;) 
            {
                strides[i] = stride;
                stride *= shape[i];
            }
        }

        size_t compute_index(const std::vector<size_t> & indices) const 
        {
            if (indices.size() != shape.size()) 
            {
                throw std::out_of_range("Tensor::compute_index: Wrong number of indices.");
            }

            size_t index = 0;
            for (size_t i = 0; i < shape.size(); ++i) 
            {
                if (indices[i] >= shape[i]) 
                {
                    std::string shape_str = "[";
                    for (size_t i = 0; i < shape.size(); ++i) shape_str += std::to_string(shape[i]) + (i + 1 == shape.size() ? "" : ", ");
                    shape_str += "]";
                    
                    throw std::out_of_range("Tensor::compute_index: " + std::to_string(indices[i]) + 
                                            "th index (" + std::to_string(i) + "th position)" + 
                                            " is out of range. The tensor shape: " + shape_str);
                }

                index += indices[i] * strides[i];
            }

            return index;
        }
    };
}