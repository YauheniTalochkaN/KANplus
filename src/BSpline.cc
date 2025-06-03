#include "BSpline.hh"

BSpline::BSpline(size_t r) : degree(r)
{

}

BSpline::~BSpline()
{
    Clear();
}

void BSpline::Initialization(const std::vector<std::pair<double, double>>& nodes)
{
    Clear();

    if(nodes.size() < 2)
    {
        throw std::invalid_argument("BSpline::Initialization: At least two nodes are required for spline size_terpolation.");
    }

    size_t n = nodes.size();
    x.resize(n);
    y.resize(n);

    for(size_t i = 0; i < n; ++i)
    {
        x[i] = nodes[i].first;
        y[i] = nodes[i].second;
    }

    knots.resize(n + degree + 1);
    
    for(size_t i = 0; i <= degree; ++i) 
        knots[i] = x.front();
    
    for(size_t i = 1; i < n - degree; ++i) 
    {
        double sum = 0;
        for(size_t j = 0; j < degree; ++j) 
        {
            sum += x[i+j];
        }

        knots[i + degree] = sum / ((degree > 0) ? (double)degree : 1.0);
    }
    
    for(size_t i = n; i <= n + degree; ++i) 
        knots[i] = x.back();

    /* for(size_t i = 0; i < n + degree + 1; ++i) std::cout << knots[i] << "\t";
    std::cout << "\n"; */
}

double BSpline::BasisFunction(const size_t & i, const size_t & k, const double & val) const
{    
    if(k == 0)
    {
        if((val == GetXmin()) && (i == degree)) return 1.0;
        else if((val == GetXmax()) && (i == x.size() - 1)) return 1.0;
        else return (val >= knots[i] && val < knots[i + 1]) ? 1.0 : 0.0;
    }

    double denom1 = knots[i + k] - knots[i];
    double denom2 = knots[i + k + 1] - knots[i + 1];

    double term1 = denom1 != 0 ? ((val - knots[i]) / denom1) * BasisFunction(i, k - 1, val) : 0;
    double term2 = denom2 != 0 ? ((knots[i + k + 1] - val) / denom2) * BasisFunction(i + 1, k - 1, val) : 0;

    return term1 + term2;
}

double BSpline::Evaluate(const double & val) const
{
    if((x.empty()) || (y.empty()))
    {
        throw std::runtime_error("BSpline::Evaluate: Spline is not initialized.");
    }

    if((val < GetXmin()) || (val > GetXmax()))
    {
        std::cout << "BSpline::Evaluate: Out of range.\n";
        
        return 0;
    }

    double result = 0.0;
    
    for(size_t i = 0; i < y.size(); ++i)
    {
        result += y[i] * BasisFunction(i, degree, val);
    }

    return result;
}

double BSpline::dBSpline_dyi(const size_t & i, const double & val) const
{
    if((x.empty()) || (y.empty()))
    {
        throw std::runtime_error("BSpline::dBSpline_dyi: Spline is not initialized.");
    }

    if((val < GetXmin()) || (val > GetXmax()))
    {
        std::cout << "BSpline::dBSpline_dyi: Out of range.\n";
        
        return 0;
    }
   
    return BasisFunction(i, degree, val);
}

double BSpline::BasisFunctionDerivative(const size_t & i, const size_t & k, const size_t & n, const double & val) const 
{
    if (n == 0)
    {
        return BasisFunction(i, k, val);
    }
    if (k == 0 || n > k) 
    {
        return 0.0;
    }

    double denom1 = knots[i + k] - knots[i];
    double denom2 = knots[i + k + 1] - knots[i + 1];

    double term1 = denom1 != 0 ? BasisFunctionDerivative(i, k - 1, n - 1, val) / denom1 : 0;
    double term2 = denom2 != 0 ? BasisFunctionDerivative(i + 1, k - 1, n - 1, val) / denom2 : 0;

    return k * (term1 - term2);
}


double BSpline::EvaluateDerivative(const double& val) const
{
    if((x.empty()) || (y.empty()))
    {
        throw std::runtime_error("BSpline::EvaluateDerivative: Spline is not initialized.");
    }

    if((val < GetXmin()) || (val > GetXmax()))
    {
        std::cout << "BSpline::EvaluateDerivative: Out of range.\n";
        
        return 0;
    }
    
    double result = 0.0;

    for(size_t i = 0; i < y.size(); ++i)
    {
        result += y[i] * BasisFunctionDerivative(i, degree, 1, val);
    }

    return result;
}

void BSpline::Clear()
{
    x.clear();
    y.clear();
    knots.clear();
}

std::vector<std::pair<double, double>> BSpline::GetNodes() const
{
    if((x.empty()) || (y.empty()))
    {
        throw std::runtime_error("BSpline::GetNodes: Spline is not initialized.");
    }

    std::vector<std::pair<double, double>> res(x.size());
    
    for(size_t i = 0; i < x.size(); ++i)
    {
        res[i] = {x[i], y[i]};
    }
    
    return res;
}

double BSpline::GetXmin() const
{
    if(x.empty())
    {
        throw std::runtime_error("BSpline::GetXmin: Spline is not initialized.");
    }

    return x.front();
}

double BSpline::GetXmax() const
{
    if(x.empty())
    {
        throw std::runtime_error("BSpline::GetXmax: Spline is not initialized.");
    }

    return x.back();
}

void BSpline::CorrectY(const std::vector<double> & vec, bool full)
{
    if(y.empty())
    {
        throw std::runtime_error("BSpline::CorrectY: Spline is not initialized.");
    }

    if(vec.empty())
    {
        throw std::runtime_error("BSpline::CorrectY: Correction vector is empty.");
    }

    if(y.size() != vec.size())
    {
        throw std::runtime_error("BSpline::CorrectY: The sizes of y and dy is not equal.");
    }

    for(size_t i = 0; i < y.size(); ++i)
    {
        if(full) 
        {
            y[i] = vec[i];
        }
        else
        {
            y[i] += vec[i];
        }
    }   
}