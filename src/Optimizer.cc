#include "Optimizer.hh"

Optimizer::Optimizer()
{

}

Optimizer::~Optimizer()
{

}

void Optimizer::SetSizes(const size_t & neu_num, const size_t & node_num)
{
    neuron_number = neu_num;
    node_number = node_num;
}