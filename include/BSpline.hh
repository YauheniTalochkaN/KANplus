#pragma once

#include <iostream>
#include <vector>
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <cmath>

class BSpline
{
public:
    BSpline(size_t);
    ~BSpline();
    void Initialization(const std::vector<std::pair<double, double>> &);
    double BasisFunction(const size_t &, const size_t &, const double &) const;
    double Evaluate(const double &) const;
    double BasisFunctionDerivative(const size_t &, const size_t &, const size_t &, const double &) const;
    double EvaluateDerivative(const double &) const;
    double dBSpline_dyi(const size_t &, const double &) const;
    void Clear();
    std::vector<std::pair<double, double>> GetNodes() const;
    double GetXmin() const;
    double GetXmax() const;
    void CorrectY(const std::vector<double> &, bool full = false);

private:
    std::vector<double> x, y, knots;
    size_t degree;
};