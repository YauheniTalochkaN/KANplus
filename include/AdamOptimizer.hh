#pragma once

#include "Optimizer.hh"
#include "Tensor.hh"

class AdamOptimizer : public Optimizer
{
public:
    AdamOptimizer(const double & lrate = 0.001, 
                  const double & b1 = 0.9, const double & b2 = 0.999, 
                  const double & o = 1.0E-8, const double & g = 0.1);
    ~AdamOptimizer() override;
    void Initialization() override;
    void Optimize(std::vector<Neuron> &) override;
    void Clear() override;

private:
    KAN::Tensor<double> deltaY, loss_ygrad, sqr_loss_ygrad;
    std::vector<size_t> optimizer_steps;
    double learning_rate, beta1, beta2, eps, gamma;
};