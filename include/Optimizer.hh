#pragma once

#include <iostream>
#include <vector>

#include "Neuron.hh"

class Optimizer
{
public:
    Optimizer();
    virtual ~Optimizer();
    void SetSizes(const size_t &, const size_t &);
    virtual void Initialization() = 0;
    virtual void Optimize(std::vector<Neuron> &) = 0;
    virtual void Clear() = 0;

protected:
    size_t neuron_number, node_number;
};