#include "AdamOptimizer.hh"

AdamOptimizer::AdamOptimizer(const double & lrate, const double & b1, const double & b2, 
                             const double & o, const double & g) : Optimizer(), 
                                                                   learning_rate(lrate), 
                                                                   beta1(b1), beta2(b2), 
                                                                   eps(o), gamma(g)
{

}

AdamOptimizer::~AdamOptimizer()
{
    Clear();
}

void AdamOptimizer::Initialization()
{
    Clear();
    
    deltaY = KAN::Tensor<double>({neuron_number, node_number}, 0); 
    loss_ygrad = KAN::Tensor<double>({neuron_number, node_number}, 0);
    sqr_loss_ygrad = KAN::Tensor<double>({neuron_number, node_number}, 0);
    optimizer_steps.resize(neuron_number, 0);
}

void AdamOptimizer::Clear()
{
    deltaY.clear();
    loss_ygrad.clear();
    sqr_loss_ygrad.clear();
    optimizer_steps.clear();
}

void AdamOptimizer::Optimize(std::vector<Neuron> & net)
{
    #pragma omp parallel for
    for(size_t i = 0; i < net.size(); ++i)
    {
        try
        {
            if(!net[i].GetDrop())
            {
                ++optimizer_steps.at(i);

                size_t opt_steps = optimizer_steps.at(i);

                KAN::Tensor<double> BasicGrads = net[i].GetBasicGrads();
                size_t batch_size = net[i].GetOutput().size();
                std::vector<double> dY_new, LossYgrad;

                LossYgrad.resize(node_number, 0);
                dY_new.resize(node_number, 0);

                for(size_t j = 0; j < node_number; ++j)
                {   
                    double sum = 0;

                    for(size_t k = 0; k < batch_size; ++k)
                    {
                        sum += BasicGrads({k, j}) * BasicGrads({k, node_number + 1}); 
                    }

                    LossYgrad[j] = sum / (double)batch_size;
                }

                net[i].FillBasicGradswithZero();

                for(size_t j = 0; j < node_number; ++j)
                {   
                    sqr_loss_ygrad({i, j}) = beta2 * sqr_loss_ygrad({i, j}) + (1.0 - beta2) * pow(LossYgrad[j], 2.0);

                    loss_ygrad({i, j}) = beta1 * loss_ygrad({i, j}) + (1.0 - beta1) * LossYgrad[j];

                    dY_new[j] = -learning_rate * loss_ygrad({i, j}) / (1.0 - pow(beta1, (double)opt_steps)) / 
                                                 (sqrt(sqr_loss_ygrad({i, j}) / (1.0 - pow(beta2, (double)opt_steps))) + eps) + 
                                                 gamma * deltaY({i, j});

                    deltaY({i, j}) = dY_new[j];
                }

                net[i].CorrectYnodes(dY_new);

                dY_new.clear();
                LossYgrad.clear();
                BasicGrads.clear();
            }
        }
        catch (const std::exception& exc) 
        {
            std::cout << "Error: AdamOptimizer::Optimize: " << exc.what() << std::endl;

            exit(EXIT_FAILURE);
        }
    }
}