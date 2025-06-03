#include "KAnetwork.hh"
#include "AdamOptimizer.hh"

double RMS(const std::vector<double> & pred, const std::vector<double> & target) 
{
    if(pred.size() != target.size()) 
    {
        throw std::invalid_argument("RMS: The sizes of pred and target must be the same!");
    }

    double total_loss = 0.0;
    size_t n = pred.size();

    for(size_t i = 0; i < n; ++i) 
    {
        total_loss += pow(pred.at(i) - target.at(i), 2.0);
    }

    return total_loss / (double)n;
}

double RMS_grad(const std::vector<double> & pred, const std::vector<double> & target, const size_t & index) 
{
    if(pred.size() != target.size()) 
    {
        throw std::invalid_argument("RMS_grad: The sizes of pred and target must be the same!");
    }
    
    return 2.0 * (pred.at(index) - target.at(index)) / (double)pred.size();
}

int main(int argc, char const* argv[])
{           
    size_t num_train_packs = 50000,
           num_in_channels = 20,
           num_out_channels = 4,
           num_test_packs = 5000;
    
    KAN::Tensor<double> in_train_data({num_train_packs, num_in_channels}), out_train_data({num_train_packs, num_out_channels}),
                        in_test_data({num_test_packs, num_in_channels}), out_test_data({num_test_packs, num_out_channels});

    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_int_distribution<> distribution_int(1, num_out_channels);
    std::uniform_real_distribution<double> distribution_double1(0.2, 0.8);
    std::uniform_real_distribution<double> distribution_double2(0.05, 0.08);
    std::uniform_real_distribution<double> distribution_double3(0.3, 1.0);
    
    for(size_t i = 0; i < num_train_packs; ++i)
    {       
        size_t num_particles = distribution_int(gen);

        std::vector<double> mu(num_particles);
        for(auto &it : mu)
        {
            it = distribution_double1(gen);
        }

        std::vector<double> sigma(num_particles);
        for(auto &it : sigma)
        {
            it = distribution_double2(gen);
        }

        std::vector<double> amp(num_particles);
        for(auto &it : amp)
        {
            it = distribution_double3(gen);
        }

        for(size_t l = 0; l < num_in_channels; ++l)
        {
            double sum = 0;
            for(size_t j = 0; j < num_particles; ++j)
            {
                sum += amp.at(j) * exp(-pow((double)l / ((double)num_in_channels - 1.0) - mu.at(j), 2.0)/(2.0 * pow(sigma.at(j), 2.0)));
            }

            in_train_data({i, l}) = sum;
        }

        std::sort(mu.begin(), mu.end());
        
        while(mu.size() < num_out_channels) 
        {
            mu.push_back(1.0);
        }

        for(size_t j = 0; j < num_out_channels; ++j)
        {
            out_train_data({i, j}) = mu.at(j);
        }

        mu.clear();
        sigma.clear();
        amp.clear();
    }
    
    for(size_t i = 0; i < num_test_packs; ++i)
    {       
        size_t num_particles = distribution_int(gen);

        std::vector<double> mu(num_particles);
        for(auto &it : mu)
        {
            it = distribution_double1(gen);
        }

        std::vector<double> sigma(num_particles);
        for(auto &it : sigma)
        {
            it = distribution_double2(gen);
        }

        std::vector<double> amp(num_particles);
        for(auto &it : amp)
        {
            it = distribution_double3(gen);
        }

        for(size_t l = 0; l < num_in_channels; ++l)
        {
            double sum = 0;
            for(size_t j = 0; j < num_particles; ++j)
            {
                sum += amp.at(j) * exp(-pow((double)l / ((double)num_in_channels - 1.0) - mu.at(j), 2.0)/(2.0 * pow(sigma.at(j), 2.0)));
            }

            in_test_data({i, l}) = sum;
        }

        std::sort(mu.begin(), mu.end());
        
        while(mu.size() < num_out_channels) 
        {
            mu.push_back(1.0);
        }

        for(size_t j = 0; j < num_out_channels; ++j)
        {
            out_test_data({i, j}) = mu.at(j);
        }

        mu.clear();
        sigma.clear();
        amp.clear();
    }

    size_t batch_size = 1000,
           max_epoch_num = 15;

    double learning_rate = 1.0E-3,
           decay_rate = 0.996,
           min_learning_rate = 1.0E-4,
           dropout = 0.05;

    /* auto end_activation = [](double & val) {if (val < 0) val = 0;};
    auto end_activation_grad = [](const double & val) -> double {if (val < 0) {return 0;} else {return 1.0;}}; */

    KAnetwork KAN(20, {0.0, 4.0}, {-1.0, 1.0}, {-1.0E-4, 1.0E-4}, {num_in_channels, 50, num_out_channels}/* , end_activation, end_activation_grad */);

    KAN.SetOptimizer(new AdamOptimizer());

    KAN.SetOmpThreadNumber(20);

    //KAN.LoadModelParameters("./model/");
    
    KAN.Train(in_train_data, out_train_data,
              in_test_data, out_test_data, 
              RMS, RMS_grad, 
              batch_size, max_epoch_num,
              learning_rate, decay_rate, min_learning_rate, dropout);

    in_train_data.clear();
    out_train_data.clear();

    KAN.ClearGrads();

    //KAN.SaveNeuronFunctions();

    KAN::Tensor<double> out_data = KAN.Evaluate(in_test_data);

    std::ofstream file("./pred.txt");

    if(file.is_open()) 
    {
    
        for(size_t i = 0; i < out_test_data.get_shape().at(0); ++i)
        {   
            for(size_t j = 0; j < out_test_data.get_shape().at(1); ++j)
            {
                file << i << "\t" << out_test_data({i, j}) << "\t" << out_data({i, j}) << "\n";
            }
        }
    }

    file.close();

    in_test_data.clear(); 
    out_test_data.clear();
    out_data.clear();

    return 0;
}