#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <omp.h>

#include "BSpline.hh"
#include "Tensor.hh"

class Neuron
{ 
public:
    Neuron(const size_t &, const size_t &);
    ~Neuron();
    double Response(const double &) const;
    void Evaluate();
    void SetInput(const std::vector<double> &);
    size_t GetBatchSize() const;
    std::vector<double> GetOutput() const;
    void SetNodes(const std::vector<std::pair<double, double>> &, bool);
    size_t GetNumber() const;
    void SetInputCoupling(Neuron*);
    void SetOutputCoupling(Neuron*);
    std::vector<Neuron*> GetInputCoupling() const;
    std::vector<Neuron*> GetOutputCoupling() const;
    void SaveState(const std::string & name1 = "./neuron_nodes.txt", 
                   const std::string & name2 = "./neuron_spline.txt", 
                   const std::string & name3 = "./neuron_spline_derivative.txt") const;
    void SaveSplineNodes(const std::string & name = "./neuron_nodes.txt") const;
    void LoadSplineNodes(const std::string & name = "./neuron_nodes.txt");
    void CorrectYnodes(const std::vector<double> &, bool full = false);
    void BasicGradsInitialization(const size_t &);
    void ClearBasicGrads();
    void EvaluateBasicGrads();
    void SetLossZGrad(const std::vector<double> &);
    KAN::Tensor<double> GetBasicGrads(const std::vector<size_t> & index = {}) const;
    void FillBasicGradswithZero();
    void SetChainID(const long int &);
    long int GetChainID() const;
    void ReNorm();
    double GetXmin() const;
    double GetXmax() const;
    double GetBias() const;
    void SetDrop(bool);
    bool GetDrop() const;
    void SetScale(const double &);
    double GetScale() const;
    double GetCenter() const;
    void SetMark(const char &);
    char GetMark() const;

private:
    size_t number, points_numbers;
    long int chain_id;
    char mark;
    std::vector<double> input, output;
    BSpline response_function;
    KAN::Tensor<double> bacis_grad;
    std::vector<Neuron*> in_couplings, out_couplings;
    double center, bias, scale;
    size_t bias_element_number;
    bool drop;
};