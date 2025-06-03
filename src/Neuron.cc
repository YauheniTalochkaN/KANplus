#include "Neuron.hh"

Neuron::Neuron(const size_t & val1, const size_t & val2) : points_numbers(val2), response_function(3)
{
    number = val1;
    chain_id = -1;
    bias = 0.0;
    scale = 1.0;
    bias_element_number = 0;
    center = 0;
    drop = false;
}

Neuron::~Neuron()
{
    in_couplings.clear ();
    out_couplings.clear();
}

size_t Neuron::GetNumber() const
{
    return number;
}

void Neuron::SetInputCoupling(Neuron* neu)
{
    in_couplings.push_back(neu);
}

void Neuron::SetOutputCoupling(Neuron* neu)
{
    out_couplings.push_back(neu);
}

std::vector<Neuron*> Neuron::GetInputCoupling() const
{
    return in_couplings;
}

std::vector<Neuron*> Neuron::GetOutputCoupling() const
{
    return out_couplings;
}

void Neuron::SetDrop(bool val)
{
    drop = val;
}

bool Neuron::GetDrop() const
{
    return drop;
}

void Neuron::SetMark(const char & name)
{
    mark = name;
}

char Neuron::GetMark() const
{
    return mark;
}

void Neuron::SetNodes(const std::vector<std::pair<double, double>>& val, bool inneu)
{    
    try 
    {
        response_function.Initialization(val);
    } 
    catch (const std::exception& exc) 
    {
        std::cout << "Error: Neuron::SetNodes: " << exc.what() << std::endl;

        exit(EXIT_FAILURE);
    }

    center = 0;

    if(!inneu)
    {
        double xmin = response_function.GetXmin(),
               xmax = response_function.GetXmax();

        center = 0.5 * (xmin + xmax);
    }
}

double Neuron::Response(const double & val) const
{
    double res = -1;
    
    try 
    {
        res = response_function.Evaluate(scale * (val - bias) + center);
    } 
    catch (const std::exception& exc) 
    {
        std::cout << "Error: Neuron::Response: " << exc.what() << std::endl;

        exit(EXIT_FAILURE);
    }
    
    return res;
}

void Neuron::ReNorm()
{
    double sum = std::accumulate(input.begin(), input.end(), 0.0);

    size_t Z = bias_element_number + input.size();

    bias = (bias * (double)bias_element_number + sum) / (double)Z;
    
    bias_element_number = Z;
}

double Neuron::GetBias() const
{
    return bias;
}

void Neuron::SetScale(const double & val)
{
    if(val <= 0.0)
    {
        throw std::invalid_argument("Neuron::SetScale: Wrong value for the spline scale factor.");
    }

    scale = val;
}

double Neuron::GetScale() const
{
    return scale;
}

double Neuron::GetXmin() const
{
    double res = 0;

    try
    {
        res = response_function.GetXmin();
    }
    catch (const std::exception& exc) 
    {
        std::cout << "Error: Neuron::GetXmin: " << exc.what() << std::endl;

        exit(EXIT_FAILURE);
    } 

    return res;   
}

double Neuron::GetXmax() const
{
    double res = 0;

    try
    {
        res = response_function.GetXmax();
    }
    catch (const std::exception& exc) 
    {
        std::cout << "Error: Neuron::GetXmax: " << exc.what() << std::endl;

        exit(EXIT_FAILURE);
    } 

    return res; 
}

void Neuron::Evaluate()
{
    output.clear();
    output.resize(input.size(), 0);
    
    #pragma omp parallel for
    for(size_t i = 0; i < input.size(); ++i)
    {
        output[i] = Response(input[i]);
    }
}

void Neuron::SetInput(const std::vector<double> & val)
{
    input.clear();

    input = val;
}

std::vector<double> Neuron::GetOutput() const
{
    return output;
}

size_t Neuron::GetBatchSize() const
{
    return input.size();
}

void Neuron::SaveState(const std::string &name1, const std::string & name2, const std::string & name3) const
{
    std::filesystem::path dirPath1 = std::filesystem::path(name1).parent_path(),
                          dirPath2 = std::filesystem::path(name2).parent_path(),
                          dirPath3 = std::filesystem::path(name3).parent_path();
    
    if (!std::filesystem::exists(dirPath1)) 
    {
        if (!std::filesystem::create_directories(dirPath1)) 
        {
            throw std::runtime_error("Neuron::SaveState: Fail to create the folder " + dirPath1.string());
        }
    }

    if (!std::filesystem::exists(dirPath2)) 
    {
        if (!std::filesystem::create_directories(dirPath2)) 
        {
            throw std::runtime_error("Neuron::SaveState: Fail to create the folder " + dirPath2.string());
        }
    }

    if (!std::filesystem::exists(dirPath3)) 
    {
        if (!std::filesystem::create_directories(dirPath3)) 
        {
            throw std::runtime_error("Neuron::SaveState: Fail to create the folder " + dirPath3.string());
        }
    }
    
    std::ofstream file1(name1), file2(name2), file3(name3);
        
    if(!file1.is_open()) 
    {
        throw std::runtime_error("Neuron::SaveState: Fail to open the file " + name1);
    }
    
    std::vector<std::pair<double, double>> nodes = response_function.GetNodes();

    double xmin = response_function.GetXmin(), 
           xmax = response_function.GetXmax();

    for(auto &it : nodes) 
    {
        file1 << (it.first - center) / scale + bias << "\t" << it.second << "\n";
    }

    file1.close();    
        
    if(!file2.is_open()) 
    {
        throw std::runtime_error("Neuron::SaveState: Fail to open the file " + name2);
    }

    for(size_t i = 0; i < 1000; ++i) 
    {
        double x = xmin + (double)i * (xmax - xmin) / 999.0;

        file2 << (x - center) / scale + bias << "\t" << response_function.Evaluate(x) << "\n";
    }

    file2.close();

    if(!file3.is_open()) 
    {
        throw std::runtime_error("Neuron::SaveState: Fail to open the file " + name3);
    }

    for(size_t i = 0; i < 1000; ++i) 
    {
        double x = xmin + (double)i * (xmax - xmin) / 999.0;

        file3 << (x - center) / scale + bias << "\t" << scale * response_function.EvaluateDerivative(x) << "\n";
    }

    file3.close();
}

void Neuron::SaveSplineNodes(const std::string & name) const
{
    std::filesystem::path dirPath = std::filesystem::path(name).parent_path();
    
    if (!std::filesystem::exists(dirPath)) 
    {
        if (!std::filesystem::create_directories(dirPath)) 
        {
            throw std::runtime_error("Neuron::SaveSplineNodes: Fail to create the folder " + dirPath.string());
        }
    }
    
    std::ofstream file(name);

    if(!file.is_open()) 
    {
        throw std::runtime_error("Neuron::SaveSplineNodes: Fail to open the file " + name);
    }

    std::vector<std::pair<double, double>> nodes = response_function.GetNodes();

    for(auto &it : nodes) 
    {
        file << (it.first - center) / scale + bias << "\t" << it.second << "\n";
    }

    file.close();
}

void Neuron::LoadSplineNodes(const std::string & name)
{
     std::ifstream file(name);

    if(!file.is_open()) 
    {
        throw std::runtime_error("Neuron::LoadSplineNodes: Fail to open the file " + name);
    }

    std::vector<std::pair<double, double>> nodes;

    std::string line;

    while (std::getline(file, line)) 
    {
        std::istringstream iss(line);
        double first, second;

        if (!(iss >> first >> second)) 
        {
            throw std::runtime_error("Neuron::LoadSplineNodes: Fail to read the file " + name);
        }

        nodes.emplace_back(first, second);
    }

    file.close();

    SetNodes(nodes, true);

    nodes.clear();
}

void Neuron::CorrectYnodes(const std::vector<double> & dy, bool full)
{
    response_function.CorrectY(dy, full);
}

void Neuron::BasicGradsInitialization(const size_t & batch_size)
{
    ClearBasicGrads();
    bacis_grad = KAN::Tensor<double>({batch_size, points_numbers + 2}, 0);
}

void Neuron::FillBasicGradswithZero()
{
    bacis_grad.set_value(0);
}

void Neuron::ClearBasicGrads()
{    
    bacis_grad.clear();
}

void Neuron::EvaluateBasicGrads()
{
    if(bacis_grad.get_shape().size() != 2)
    {
        throw std::runtime_error("Neuron::EvaluateBasicGrads: Basic gradient vector is not initiated.");
    }

    if(!(bacis_grad.get_shape()[0]) || !(bacis_grad.get_shape()[1]))
    {
        throw std::runtime_error("Neuron::EvaluateBasicGrads: Basic gradient vector is not initiated.");
    }

    #pragma omp parallel for
    for(size_t i = 0; i < input.size(); ++i)
    {
        for(size_t j = 0; j < points_numbers; ++j)
        {
            try
            {
                bacis_grad({i, j}) = response_function.dBSpline_dyi(j, scale * (input[i] - bias) + center);
            }
            catch (const std::exception& exc) 
            {
                std::cout << "Error: Neuron::EvaluateBasicGrads: " << exc.what() << std::endl;

                exit(EXIT_FAILURE);
            }
        }
    
        try
        {
            bacis_grad({i, points_numbers}) = scale * response_function.EvaluateDerivative(scale * (input[i] - bias) + center);
        }
        catch (const std::exception& exc) 
        {
            std::cout << "Error: Neuron::EvaluateBasicGrads: " << exc.what() << std::endl;

            exit(EXIT_FAILURE);
        }
    }
}

void Neuron::SetLossZGrad(const std::vector<double> & vec)
{    
    if(bacis_grad.get_shape().size() != 2)
    {
        throw std::runtime_error("Neuron::SetLossZGrad: Basic gradient vector is not initiated.");
    }

    if(!(bacis_grad.get_shape()[0]) || !(bacis_grad.get_shape()[1]))
    {
        throw std::runtime_error("Neuron::SetLossZGrad: Basic gradient vector is not initiated.");
    }

    if(input.size() != vec.size())
    {
        throw std::runtime_error("Neuron::SetLossZGrad: The size of new dL/dZ vector is wrong.");
    }
    
    for(size_t i = 0; i < vec.size(); ++i)
    {
        try
        {
            bacis_grad({i, points_numbers + 1}) = vec[i];
        }
        catch (const std::exception& exc) 
        {
            std::cout << "Error: Neuron::SetLossZGrad: " << exc.what() << std::endl;

            exit(EXIT_FAILURE);
        }
    }
}

void Neuron::SetChainID(const long int & k)
{
    chain_id = k;
}

KAN::Tensor<double> Neuron::GetBasicGrads(const std::vector<size_t> & index) const
{
    if(bacis_grad.get_shape().size() != 2)
    {
        throw std::runtime_error("Neuron::GetBasicGrads: Basic gradient vector is empty.");
    }

    if(!(bacis_grad.get_shape()[0]) || !(bacis_grad.get_shape()[1]))
    {
        throw std::runtime_error("Neuron::GetBasicGrads: Basic gradient vector is empty.");
    }

    if(index.size() > 0)
    {
        KAN::Tensor<double> data({input.size(), index.size()}, 0);

        for(size_t i = 0; i < input.size(); ++i)
        {
            for(size_t j = 0; j < index.size(); ++j)
            {
                if(index[j] >= 0 && index[j] <= points_numbers + 1)
                {
                    data({i, j}) = bacis_grad({i, index[j]});
                }
                else
                {
                    throw std::out_of_range("Neuron::GetBasicGrads: Wrong index of gradient element.");
                }
            }
        }

        return data;   
    }
    else
    {
        return bacis_grad;
    }
}

long int Neuron::GetChainID() const
{
    return chain_id;
}

double Neuron::GetCenter() const
{
    return center;
}