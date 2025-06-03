#pragma once

#include <iostream>
#include <bits/stdc++.h>
#include <ctime>
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm> 
#include <utility>
#include <array>
#include <stdexcept>
#include <functional>
#include <string>
#include <cctype> 

#include "Neuron.hh"
#include "Tensor.hh"
#include "Optimizer.hh"

class KAnetwork
{ 
public:
    KAnetwork(const size_t &, const std::pair<double, double> &, 
              const std::pair<double, double> &, const std::pair<double, double> &, 
              const size_t &, const std::vector<std::pair<size_t, size_t>> &, 
              const std::vector<std::vector<size_t>> &, const std::vector<std::vector<size_t>> &, 
              const std::function<void(double &)> & final_activation = nullptr, 
              const std::function<double(const double &)> & final_activation_grad = nullptr);
    KAnetwork(const size_t &, 
              const std::pair<double, double> &, const std::pair<double, double> &, const std::pair<double, double> &,
              const std::vector<size_t> &, 
              const std::function<void(double &)> & final_activation = nullptr, 
              const std::function<double(const double &)> & final_activation_grad = nullptr);
    ~KAnetwork();
    void Initialization(const size_t &, const std::pair<double, double> &, const std::pair<double, double> &, const std::pair<double, double> &, 
                        const size_t &, const std::vector<std::pair<size_t, size_t>> &, 
                        const std::vector<std::vector<size_t>> &, const std::vector<std::vector<size_t>> &, 
                        const std::function<void(double &)> & final_activation = nullptr, 
                        const std::function<double(const double &)> & final_activation_grad = nullptr);
    size_t CreateModelGraph(const std::vector<size_t> &, std::vector<std::pair<size_t, size_t>> &, 
                                  std::vector<std::vector<size_t>> &, std::vector<std::vector<size_t>> &) const;
    void SetOmpThreadNumber(const size_t &);
    void SetOptimizer(Optimizer*);
    void Forward();
    void SetInputs(const KAN::Tensor<double> &);
    KAN::Tensor<double> GetOutput();
    void Backpropagation(const KAN::Tensor<double> &, const KAN::Tensor<double> &,
                         const std::function<double(const std::vector<double> &, const std::vector<double> &, const size_t &)> &);
    void Train(const KAN::Tensor<double> &, const KAN::Tensor<double> &,
               const KAN::Tensor<double> &, const KAN::Tensor<double> &,
               const std::function<double(const std::vector<double> &, const std::vector<double> &)> &,
               const std::function<double(const std::vector<double> &, const std::vector<double> &, const size_t &)> &, 
               size_t &, size_t &, const double & learning_rate = 0.001, const double & decay_rate = 1, const double & min_learning_rate = 1.0E-6, double dropout = 0,
               const std::function<void(const std::vector<double> &, const std::vector<double> &)> & test_func = nullptr);
    void SetMode(std::string);
    void SetSplineXrange(const std::pair<double, double> &, const std::pair<double, double> &);
    KAN::Tensor<double> Evaluate(KAN::Tensor<double>);
    void ClearGrads();
    void ClearModel();
    void SaveNeuronFunctions(const std::string & prefix = "./neurons_data/") const;
    void SaveModelParameters(const std::string & prefix = "./model/") const;
    void LoadModelParameters(const std::string & prefix = "./model/");
    void SetDropOut(double);
    void ResetDropOut();

private:
    std::vector<Neuron> net;
    Optimizer* optimizer = nullptr;
    std::vector<Neuron*> input_neurons, output_neurons;
    std::vector<std::vector<Neuron*>> propagation_chain;
    std::vector<size_t> in_channels, out_channels;
    size_t spline_points_numbers;
    std::function<void(double &)> end_activation;
    std::function<double(const double &)> end_activation_grad;
    double xmin1, xmax1, xmin2, xmax2;
    size_t thread_num;
    size_t num_out_channels;
    bool train;
    double dropout;
};