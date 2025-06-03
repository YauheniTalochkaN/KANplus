#include "KAnetwork.hh"

KAnetwork::KAnetwork(const size_t & node_num, 
                     const std::pair<double, double> & xrange1, const std::pair<double, double> & xrange2, const std::pair<double, double> & yrange,
                     const size_t & neuron_num, const std::vector<std::pair<size_t, size_t>> & graph, 
                     const std::vector<std::vector<size_t>> & in_neu, const std::vector<std::vector<size_t>> & out_neu, 
                     const std::function<void(double &)> & final_activation, 
                     const std::function<double(const double &)> & final_activation_grad)
{
    train = false;

    SetOmpThreadNumber(1);

    dropout = 0.0;

    try 
    {
        Initialization(node_num, xrange1, xrange2, yrange, neuron_num, graph, in_neu, out_neu, final_activation, final_activation_grad);
    } 
    catch (const std::exception& exc) 
    {
        std::cout << "Error: KAnetwork::KAnetwork: " << exc.what() << std::endl;

        exit(EXIT_FAILURE);
    }
}

KAnetwork::KAnetwork(const size_t & node_num, 
                     const std::pair<double, double> & xrange1, const std::pair<double, double> & xrange2, const std::pair<double, double> & yrange,
                     const std::vector<size_t> & layers, 
                     const std::function<void(double &)> & final_activation, 
                     const std::function<double(const double &)> & final_activation_grad)
{
    train = false;

    SetOmpThreadNumber(1);

    dropout = 0.0;

    std::vector<std::pair<size_t, size_t>> graph;
    std::vector<std::vector<size_t>> in_neu, out_neu;

    size_t neuron_num = 0;

    try 
    {
        neuron_num = CreateModelGraph(layers, graph, in_neu, out_neu);
        
        Initialization(node_num, xrange1, xrange2, yrange, neuron_num, graph, in_neu, out_neu, final_activation, final_activation_grad);
    } 
    catch (const std::exception& exc) 
    {
        std::cout << "Error: KAnetwork::KAnetwork: " << exc.what() << std::endl;

        exit(EXIT_FAILURE);
    }
}

KAnetwork::~KAnetwork()
{
    ClearModel();

    if(optimizer != nullptr) delete optimizer;
}

void KAnetwork::SetOmpThreadNumber(const size_t & val)
{
    if(val < 1)
    {
        throw std::invalid_argument("KAnetwork::SetOmpThreadNumber: Wrong number of omp threads.");
    }
    
    thread_num = val;
    
    omp_set_num_threads(thread_num);
}

void KAnetwork::ClearModel()
{
    ClearGrads();
    if(optimizer != nullptr) optimizer->Clear();
    input_neurons.clear();
    output_neurons.clear();
    net.clear();
}

void KAnetwork::SetMode(std::string str)
{
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
    
    if(str == "train") {train = true;}
    else {train = false;}

    ResetDropOut();
}

void KAnetwork::SetSplineXrange(const std::pair<double, double> & xrange1, const std::pair<double, double> & xrange2)
{
    if(xrange1.first >= xrange1.second)
    {
        throw std::invalid_argument("KAnetwork::SetSplineXrange: Wrong values for xmin and xmax.");
    }

    xmin1 = xrange1.first;
    xmax1 = xrange1.second;

    if(xrange2.first >= xrange2.second)
    {
        throw std::invalid_argument("KAnetwork::SetSplineXrange: Wrong values for xmin and xmax.");
    }

    xmin2 = xrange2.first;
    xmax2 = xrange2.second;
}

void KAnetwork::Initialization(const size_t & node_num, const std::pair<double, double> & xrange1, const std::pair<double, double> & xrange2, const std::pair<double, double> & yrange,
                               const size_t & neuron_num, const std::vector<std::pair<size_t, size_t>> & graph, 
                               const std::vector<std::vector<size_t>> & in_neu, const std::vector<std::vector<size_t>> & out_neu, 
                               const std::function<void(double &)> & final_activation, 
                               const std::function<double(const double &)> & final_activation_grad)
{
    ClearModel();
    
    end_activation = final_activation;
    end_activation_grad = final_activation_grad;
    
    spline_points_numbers = 2;
    if(node_num > 2) spline_points_numbers = node_num;

    try
    {
        SetSplineXrange(xrange1, xrange2);
    }
    catch (const std::exception& exc) 
    {
        std::cout << "Error: " << exc.what() << std::endl;

        exit(EXIT_FAILURE);
    }

    double ymin = std::min(xmin1/1000.0, xmin2/1000.0), 
           ymax = std::min(xmax1/1000.0, xmax2/1000.0);
    
    if(yrange.first < yrange.second)
    {
        ymin = yrange.first;
        ymax = yrange.second;
    }

    if(neuron_num > 0)
    { 
        net.reserve(neuron_num);

        for(std::size_t i = 0; i < neuron_num; ++i)
        {
            Neuron neu(i, spline_points_numbers);

            net.push_back(neu);
        }
    }
    else 
    {
        throw std::invalid_argument("KAnetwork::Initialization: Wrong number of neurons.");
    }
    
    for(auto &it : graph)
    {
        if((it.first >= 0) && (it.first < net.size()) && (it.second >= 0) && (it.second < net.size()))
        {
            net[it.first].SetOutputCoupling(&net[it.second]); 
            net[it.second].SetInputCoupling(&net[it.first]);  
        }
        else
        {       
            throw std::invalid_argument("KAnetwork::Initialization: Wrong numbers of neurons in the graph.");
        }
    }

    for(size_t i = 0; i < in_neu.size(); ++i)
    {
        for(auto &it : in_neu[i])
        {
            try
            {
                input_neurons.push_back(&net.at(it));
                in_channels.push_back(i);
                net.at(it).SetMark('i');
            }
            catch (std::out_of_range& exc)
            {
                std::cout << "Error: KAnetwork::Initialization: Incorrect index in the vector of the numbers of input neurons.\n";

                exit(EXIT_FAILURE);
            }
        }
    }

    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<double> distribution(ymin, ymax);

    for(auto &it : net)
    {
        auto found = std::find_if(input_neurons.begin(), input_neurons.end(),
                                 [&it](Neuron* obj) -> bool {return obj->GetNumber() == it.GetNumber();});

        double xmin = xmin1, xmax = xmax1;
        bool inneu = true;

        if(found == input_neurons.end())
        {
            xmin = xmin2; 
            xmax = xmax2;
            inneu = false;
        }   
        
        std::vector<std::pair<double, double>> nodes;
        nodes.resize(spline_points_numbers);

        for(std::size_t j = 0; j < spline_points_numbers; ++j)
        {            
            nodes[j] = std::make_pair(xmin + (xmax - xmin) * ((double)j / (double)(spline_points_numbers-1)), distribution(gen));
        }

        it.SetNodes(nodes, inneu); 

        nodes.clear();
    }

    num_out_channels = out_neu.size();

    for(size_t i = 0; i < out_neu.size(); ++i)
    {
        for(auto &it : out_neu[i])
        {
            try
            {
                output_neurons.push_back(&net.at(it));
                out_channels.push_back(i);
                net.at(it).SetMark('o');
            }
            catch (std::out_of_range& exc)
            {
                std::cout << "Error: KAnetwork::Initialization: Incorrect index in the vector of the numbers of output neurons.\n";

                exit(EXIT_FAILURE);
            }
        }
    }

    /* for(auto &it1 : net)
    {
        if(it1.GetInputCoupling().size())
        {
            for(auto &it2 : it1.GetInputCoupling())
            {
                std::cout << "Neuron number: " << it1.GetNumber() << "\tInput neuron number: " << it2->GetNumber() << "\n";
            }
        }
    } */

    /* for(auto &it1 : net)
    {
        if(it1.GetOutputCoupling().size())
        {
            for(auto &it2 : it1.GetOutputCoupling())
            {
                std::cout << "Neuron number: " << it1.GetNumber() << "\tOutput neuron number: " << it2->GetNumber() << "\n";
            }
        }
    } */

    std::vector<Neuron*> pre_neu_vec = input_neurons, post_neu_vec;
    
    long int ch_id = 0;

    while(pre_neu_vec.size())
    {
        for(auto &it1 : pre_neu_vec)
        {
            bool apply = true;

            if(ch_id > 0)
            {
                std::vector<Neuron*> in = it1->GetInputCoupling();

                for(auto &it2 : in)
                {
                    if((it2->GetChainID() < 0) || (it2->GetChainID() == ch_id))
                    {
                        apply = false;
                        break;
                    }
                }

                in.clear();
            }

            if(apply)
            {
                it1->SetChainID(ch_id);

                std::vector<Neuron*> out = it1->GetOutputCoupling();

                for(auto &it2 : out)
                {
                    auto found = std::find_if(post_neu_vec.begin(), post_neu_vec.end(),
                                             [&it2](Neuron* obj) {return obj->GetNumber() == it2->GetNumber();});

                    if(found == post_neu_vec.end()) 
                    {
                        post_neu_vec.push_back(it2);
                    }
                }

                out.clear();
            }
        }

        pre_neu_vec.clear();

        pre_neu_vec = post_neu_vec;

        post_neu_vec.clear();

        ++ch_id;
    }

    pre_neu_vec.clear();

    propagation_chain.resize(ch_id);

    for(auto &it : net)
    {
        propagation_chain[it.GetChainID()].push_back(&it);
    }

    /* for(auto &it : net)
    {
        std::cout << "Neuron number: " << it.GetNumber() << "\tNeuron chain ID: " << it.GetChainID() << "\n";
    } */
}

size_t KAnetwork::CreateModelGraph(const std::vector<size_t> & layers, 
                                   std::vector<std::pair<size_t, size_t>> & graph, 
                                   std::vector<std::vector<size_t>> & in, 
                                   std::vector<std::vector<size_t>> & out) const
{
    size_t edgeIndex = 0;
    std::vector<std::vector<size_t>> layerEdges;

    in.resize(layers.front());
    out.resize(layers.back());

    for (size_t l = 0; l < layers.size() - 1; ++l) 
    {
        size_t currentLayerSize = layers[l];
        size_t nextLayerSize = layers[l + 1];

        std::vector<size_t> currentLayerEdges(currentLayerSize * nextLayerSize);

        for (size_t i = 0; i < currentLayerSize; ++i) 
        {
            for (size_t j = 0; j < nextLayerSize; ++j) 
            {
                currentLayerEdges[i * nextLayerSize + j] = edgeIndex;

                if (l == 0) 
                {
                    in[i].push_back(edgeIndex);
                }

                if (l == layers.size() - 2) 
                {
                    out[j].push_back(edgeIndex);
                }

                ++edgeIndex;
            }
        }

        layerEdges.push_back(std::move(currentLayerEdges));

        currentLayerEdges.clear();
    }

    for (size_t l = 0; l < layerEdges.size() - 1; ++l) 
    {
        const std::vector<size_t> & currentEdges = layerEdges[l];
        const std::vector<size_t> & nextEdges = layerEdges[l + 1];

        size_t currentLayerSize = layers[l + 1];

        for (size_t i = 0; i < currentLayerSize; ++i) 
        {
            for (size_t j = 0; j < layers[l]; ++j) 
            {
                size_t closerEdge = currentEdges[j * currentLayerSize + i];
                for (size_t k = 0; k < layers[l + 2]; ++k) 
                {
                    size_t fartherEdge = nextEdges[i * layers[l + 2] + k];
                    graph.emplace_back(closerEdge, fartherEdge);
                }
            }
        }
    }

    layerEdges.clear();

    return edgeIndex;
}

void KAnetwork::SetInputs(const KAN::Tensor<double> & in)
{        
    for(size_t i = 0; i < input_neurons.size(); ++i)
    {
        try
        {
            std::vector<double> neu_in_vector;
            neu_in_vector.resize(in.get_shape().at(0), 0);

            for(size_t j = 0; j < neu_in_vector.size(); ++j)
            {
                neu_in_vector[j] = in({j, in_channels.at(i)});
            }
            
            input_neurons[i]->SetInput(neu_in_vector);

            neu_in_vector.clear();
        }
        catch (std::out_of_range& exc)
        {
            std::cout << "Error: KAnetwork::SetInputs: " << exc.what() << std::endl;

            exit(EXIT_FAILURE);
        }
    }
}

KAN::Tensor<double> KAnetwork::GetOutput()
{
    KAN::Tensor<double> out({input_neurons.at(0)->GetBatchSize(), num_out_channels}, 0);

    for(std::size_t i = 0; i < output_neurons.size(); ++i)
    {
        if(!output_neurons[i]->GetDrop())
        {
            std::vector<double> neu_out = output_neurons[i]->GetOutput();

            for(std::size_t j = 0; j < out.get_shape()[0]; ++j)
            {
                try
                {
                    out({j, out_channels.at(i)}) += neu_out.at(j);
                }
                catch (std::out_of_range& exc)
                {
                    std::cout << "Error: KAnetwork::GetOutput: " << exc.what() << std::endl;

                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    return out;
}

void KAnetwork::Forward()
{
    size_t batch_size = input_neurons.at(0)->GetBatchSize();

    for(auto &neu_vec : propagation_chain)
    {      
        #pragma omp parallel for
        for(size_t i = 0; i < neu_vec.size(); ++i)
        {
            if(!neu_vec[i]->GetDrop())
            {
                if(neu_vec[i]->GetMark() != 'i')
                {
                    std::vector<Neuron*> in = neu_vec[i]->GetInputCoupling();

                    std::vector<double> signal;
                    signal.resize(batch_size, 0);

                    for(auto &it : in)
                    {
                        if(!it->GetDrop())
                        {
                            std::vector<double> local_out = it->GetOutput();

                            std::transform(signal.begin(), signal.end(), local_out.begin(), signal.begin(), 
                                          [](double a, double b) -> double {return a + b;});
                        }
                    }

                    in.clear();

                    if((dropout > 0.0) && (dropout < 1.0))
                    {
                        for(auto &it : signal)
                        {
                            it /= 1.0 - dropout;
                        }
                    }

                    neu_vec[i]->SetInput(signal);

                    if(train)
                    {
                        neu_vec[i]->ReNorm();

                        auto [min_it, max_it] = std::minmax_element(signal.begin(), signal.end());

                        double neu_xmin = neu_vec[i]->GetXmin(),
                               neu_xmax = neu_vec[i]->GetXmax(),
                               neu_bias = neu_vec[i]->GetBias(),
                               neu_scale = neu_vec[i]->GetScale(),
                               neu_center = neu_vec[i]->GetCenter();

                        if(((neu_scale * ((*min_it) - neu_bias) + neu_center) < neu_xmin) ||
                           ((neu_scale * ((*max_it) - neu_bias) + neu_center) > neu_xmax))
                        {
                           neu_vec[i]->SetScale(0.5 * neu_vec[i]->GetScale());
                        }
                    }

                    signal.clear();
                }

                neu_vec[i]->Evaluate();

                if(train)
                {
                    try
                    {
                        neu_vec[i]->EvaluateBasicGrads();
                    }
                    catch (const std::exception& exc) 
                    {
                        std::cout << "Error: KAnetwork::Forward: " << exc.what() << std::endl;
                        
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
    }    
}

void KAnetwork::Backpropagation(const KAN::Tensor<double> & res, const KAN::Tensor<double> & answer,
                                const std::function<double(const std::vector<double> &, const std::vector<double> &, const size_t &)> & loss_function_grad)
{
    size_t batch_size = res.get_shape().at(0);
    
    #pragma omp parallel for
    for(size_t i = 0; i < output_neurons.size(); ++i)
    {                
        if(!output_neurons[i]->GetDrop())
        { 
            std::vector<double> local_loss_zgrad;
            local_loss_zgrad.resize(batch_size, 0);

            for(size_t j = 0; j < batch_size; ++j)
            {
                try
                {
                    if((end_activation != nullptr) && (end_activation_grad != nullptr))
                    {
                        size_t channel = out_channels.at(i);
                        
                        std::vector<double> sigma_z = res.get_slice_1d(j), 
                                            dsigmadz;

                        dsigmadz.resize(sigma_z.size(), 0);
                        
                        for(size_t k = 0; k < dsigmadz.size(); ++k)
                        {
                            dsigmadz[k] = end_activation_grad(sigma_z[k]);
                        }
                        for(auto &it : sigma_z)
                        {
                            end_activation(it);
                        }

                        local_loss_zgrad[j] = dsigmadz.at(channel) * loss_function_grad(sigma_z, answer.get_slice_1d(j), channel);
                    }
                    else
                    {
                        local_loss_zgrad[j] = loss_function_grad(res.get_slice_1d(j), answer.get_slice_1d(j), out_channels.at(i));
                    }
                }
                catch (const std::exception& exc) 
                {
                    std::cout << "Error: KAnetwork::Backpropagation: " << exc.what() << std::endl;

                    exit(EXIT_FAILURE);
                }
            }

            output_neurons[i]->SetLossZGrad(local_loss_zgrad);

            local_loss_zgrad.clear();
        }
    }
    
    for(size_t k = propagation_chain.size(); k-- > 0;)
    {
        std::vector<Neuron*> & neu_vec = propagation_chain[k];
            
        #pragma omp parallel for
        for(size_t i = 0; i < neu_vec.size(); ++i)
        {
            if((!neu_vec[i]->GetDrop()) && (neu_vec[i]->GetMark() != 'o'))
            {
                std::vector<Neuron*> out = neu_vec[i]->GetOutputCoupling();

                std::vector<double> local_loss_zgrad;
                local_loss_zgrad.resize(batch_size, 0);

                for(auto &it : out)
                {
                    if(!it->GetDrop())
                    {
                        KAN::Tensor<double> local_basic_grad;

                        try 
                        {
                            local_basic_grad = it->GetBasicGrads({spline_points_numbers, spline_points_numbers + 1});
                        } 
                        catch (const std::exception& exc) 
                        {
                            std::cout << "Error: KAnetwork::Backpropagation: " << exc.what() << std::endl;

                            exit(EXIT_FAILURE);
                        }

                        for(size_t j = 0; j < batch_size; ++j)
                        {
                            try 
                            {
                                local_loss_zgrad[j] += local_basic_grad({j, 0}) * local_basic_grad({j, 1});
                            } 
                            catch (const std::exception& exc) 
                            {
                                std::cout << "Error: KAnetwork::Backpropagation: " << exc.what() << std::endl;

                                exit(EXIT_FAILURE);
                            }
                        }

                        local_basic_grad.clear();
                    }
                }

                out.clear();

                neu_vec[i]->SetLossZGrad(local_loss_zgrad);

                local_loss_zgrad.clear();
            }
        }
    }    
}

void KAnetwork::SetDropOut(double drop_prob)
{
    if((drop_prob >= 1.0) || (drop_prob < 0.0))
    {
        throw std::invalid_argument("KAnetwork::SetDropOut: Wrong value of the dropout.");
    }
    
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<double> distribution_double(0.0, 1.0);

    for(auto &it : net)
    {
        if(distribution_double(gen) < drop_prob)
        {
            it.SetDrop(true);
        }
    }
}

void KAnetwork::ResetDropOut()
{
    for(auto &it : net)
    {
        it.SetDrop(false);
    }
}

void KAnetwork::SetOptimizer(Optimizer* opt)
{
    optimizer = opt;
}

void KAnetwork::Train(const KAN::Tensor<double> & train_data, const KAN::Tensor<double> & train_answers,
                      const KAN::Tensor<double> & test_data, const KAN::Tensor<double> & test_answers,
                      const std::function<double(const std::vector<double> &, const std::vector<double> &)> & loss_function,
                      const std::function<double(const std::vector<double> &, const std::vector<double> &, const size_t &)> & loss_function_grad, 
                      size_t & batch_size, size_t & max_epoch_num, const double & learning_rate, const double & decay_rate, const double & min_learning_rate, double drop_prob,
                      const std::function<void(const std::vector<double> &, const std::vector<double> &)> & test_func)
{    
    struct timespec global_start, global_finish;

    clock_gettime(CLOCK_MONOTONIC, &global_start);

    std::cout << "Training the model...\n";

    SetMode("train");

    if(optimizer == nullptr)
    {
        throw std::runtime_error("KAnetwork::Train: Optimizer is not set.");
    }
    
    optimizer->SetSizes(net.size(), spline_points_numbers);
    optimizer->Initialization();

    if(batch_size < 0) batch_size = 1;
    if(max_epoch_num < 0) max_epoch_num = 1;

    dropout = drop_prob;

    for(size_t i = 0; i < net.size(); ++i)
    {
        net[i].BasicGradsInitialization(batch_size);
    }

    double local_learning_rate = learning_rate;

    std::ofstream loss_file("./loss.txt");

    if(!loss_file.is_open()) 
    {
        throw std::runtime_error("KAnetwork::Train: Fail to open the loss.txt file.");
    }

    size_t iter = 1;
    while(iter <= max_epoch_num)
    {        
        KAN::Tensor<double> train_data_batch, train_answers_batch;
        std::vector<double> train_data_batch_vector, train_answers_batch_vector;
        
        size_t num_packs = train_data.get_shape().at(0);

        double loss = 0;
        size_t kloss = 0;

        try
        {
            SetDropOut(dropout);
        }
        catch (const std::exception& exc) 
        {
            std::cout << "Error: KAnetwork::Train: " << exc.what() << std::endl;

            exit(EXIT_FAILURE);
        }
        
        size_t current_batch_size = 0;
        
        for(size_t i = 0; i < num_packs; ++i)
        {
            for(size_t j = 0; j < train_data.get_shape().at(1); ++j)
            {
                train_data_batch_vector.push_back(train_data({i, j}));
            }

            for(size_t j = 0; j < train_answers.get_shape().at(1); ++j)
            {
                train_answers_batch_vector.push_back(train_answers({i, j}));
            }

            ++current_batch_size;
            
            if(((i+1) % batch_size == 0) || ((i+1) == num_packs))
            {              
                train_data_batch = KAN::Tensor<double>({current_batch_size, train_data.get_shape().at(1)}, train_data_batch_vector); 
                train_answers_batch = KAN::Tensor<double>({current_batch_size, train_answers.get_shape().at(1)}, train_answers_batch_vector);
                
                SetInputs(train_data_batch);

                Forward();

                KAN::Tensor<double> res = GetOutput();

                Backpropagation(res, train_answers_batch, loss_function_grad);

                optimizer->Optimize(net);

                if((end_activation != nullptr) && (end_activation_grad != nullptr))
                {
                    for(size_t i1 = 0; i1 < res.get_shape().at(0); ++i1)
                    {
                        for(size_t i2 = 0; i2 < res.get_shape().at(1); ++i2)
                        {
                            end_activation(res({i1, i2}));
                        }
                    }
                }

                std::vector<double> total_res, total_answers;

                for(size_t i1 = 0; i1 < res.get_shape().at(0); ++i1)
                {
                    for(size_t i2 = 0; i2 < res.get_shape().at(1); ++i2)
                    {
                        total_res.push_back(res({i1, i2}));
                    }
                }

                for(size_t i1 = 0; i1 < train_answers_batch.get_shape().at(0); ++i1)
                {
                    for(size_t i2 = 0; i2 < train_answers_batch.get_shape().at(1); ++i2)
                    {
                        total_answers.push_back(train_answers_batch({i1, i2}));
                    }
                }

                try 
                {
                    loss += loss_function(total_res, total_answers);
                } 
                catch (const std::exception& exc) 
                {
                    std::cout << "Error: KAnetwork::Train: " << exc.what() << std::endl;

                    exit(EXIT_FAILURE);
                }

                res.clear();
                train_data_batch_vector.clear();
                train_answers_batch_vector.clear();
                train_data_batch.clear(); 
                train_answers_batch.clear();
                total_res.clear();
                total_answers.clear();

                ++kloss;
                current_batch_size = 0;
            }
        }

        loss /= (double)kloss;

        std::cout << "Epoch: " << iter << "; Loss (train): " << loss << ";"; 
        loss_file << iter << "\t" << loss;

        if((test_data.get_shape().size() == 2) && (test_answers.get_shape().size() == 2)) 
        if(test_data.get_shape().at(0) == test_answers.get_shape().at(0))
        {
            SetMode("evaluate");

            SetInputs(test_data);

            Forward();

            KAN::Tensor<double> test_res = GetOutput();

            if((end_activation != nullptr) && (end_activation_grad != nullptr))
            {
                for(size_t i1 = 0; i1 < test_res.get_shape().at(0); ++i1)
                {
                    for(size_t i2 = 0; i2 < test_res.get_shape().at(1); ++i2)
                    {
                        end_activation(test_res({i1, i2}));
                    }
                }
            }

            std::vector<double> total_test_res, total_test_answers;

            for(size_t i1 = 0; i1 < test_res.get_shape().at(0); ++i1)
            {
                for(size_t i2 = 0; i2 < test_res.get_shape().at(1); ++i2)
                {
                    total_test_res.push_back(test_res({i1, i2}));
                }
            }

            for(size_t i1 = 0; i1 < test_answers.get_shape().at(0); ++i1)
            {
                for(size_t i2 = 0; i2 < test_answers.get_shape().at(1); ++i2)
                {
                    total_test_answers.push_back(test_answers({i1, i2}));
                }
            }

            double test_loss = 0;

            try 
            {
                test_loss = loss_function(total_test_res, total_test_answers);

                if(test_func != nullptr) test_func(total_test_res, total_test_answers);
            } 
            catch (const std::exception& exc) 
            {
                std::cout << "Error: KAnetwork::Train: " << exc.what() << std::endl;

                exit(EXIT_FAILURE);
            }

            test_res.clear();
            total_test_res.clear();
            total_test_answers.clear();

            std::cout << " Loss (test): " << test_loss << ";";
            loss_file << "\t" << test_loss;

            SetMode("train");
        }

        std::cout << "\n";
        loss_file << "\n";

        ++iter;

        double pred_learning_rate = local_learning_rate * decay_rate;
        
        if(pred_learning_rate > min_learning_rate) local_learning_rate = pred_learning_rate;

        SaveModelParameters();
    }

    loss_file.close();

    std::cout << "Training is done. \n";

    clock_gettime(CLOCK_MONOTONIC, &global_finish);

    printf("Total CPU time: %.5f s\n\n", (global_finish.tv_sec - global_start.tv_sec) + (global_finish.tv_nsec - global_start.tv_nsec) / 1.0E9);
}

KAN::Tensor<double> KAnetwork::Evaluate(KAN::Tensor<double> in_data)
{
    struct timespec global_start, global_finish;

    clock_gettime(CLOCK_MONOTONIC, &global_start);

    std::cout << "Evaluating the input data...\n";

    SetMode("evaluate");

    SetInputs(in_data);

    Forward();

    KAN::Tensor<double> res = GetOutput();

    if(end_activation != nullptr)
    {
        for(size_t i1 = 0; i1 < res.get_shape().at(0); ++i1)
        {
            for(size_t i2 = 0; i2 < res.get_shape().at(1); ++i2)
            {
                end_activation(res({i1, i2}));
            }
        }
    }

    std::cout << "Evaluation is done. \n";

    clock_gettime(CLOCK_MONOTONIC, &global_finish);

    printf("Total CPU time: %.5f s\n\n", (global_finish.tv_sec - global_start.tv_sec) + (global_finish.tv_nsec - global_start.tv_nsec) / 1.0E9);
        
    return res;
}

void KAnetwork::ClearGrads()
{
    for(auto &neu : net)
    {
        neu.ClearBasicGrads();
    }

    if(optimizer != nullptr) optimizer->Clear();
}

void KAnetwork::SaveNeuronFunctions(const std::string & prefix) const
{    
    std::cout << "Saving neuron functions...\n";
    
    int i = 0;
    for(auto &neu : net)
    {
        try
        {
            neu.SaveState(prefix + "neuron_nodes_" + std::to_string(i) + ".txt", 
                          prefix + "neuron_spline_" + std::to_string(i) + ".txt", 
                          prefix + "neuron_spline_derivative_" + std::to_string(i) + ".txt");
        }
        catch (const std::exception& exc) 
        {
            std::cout << "Error: KAnetwork::SaveNetState: " << exc.what() << std::endl;
        }
        
        ++i;
    }

    std::cout << "The functions are saved.\n";
}

void KAnetwork::SaveModelParameters(const std::string & prefix) const
{
    int i = 0;
    for(auto &neu : net)
    {
        try
        {
            neu.SaveSplineNodes(prefix + "neuron_nodes_" + std::to_string(i) + ".txt");
        }
        catch (const std::exception& exc) 
        {
            std::cout << "Error: KAnetwork::SaveModelParameters: " << exc.what() << std::endl;
        }
        
        ++i;
    }
}

void KAnetwork::LoadModelParameters(const std::string & prefix)
{
    std::cout << "Loading model parameters...\n";

    int i = 0;
    for(auto &neu : net)
    {
        try
        {
            neu.LoadSplineNodes(prefix + "neuron_nodes_" + std::to_string(i) + ".txt");
        }
        catch (const std::exception& exc) 
        {
            std::cout << "Error: KAnetwork::LoadModelParameters: " << exc.what() << std::endl;

            exit(EXIT_FAILURE);
        }
        
        ++i;
    }

    std::cout << "The parameters are loaded.\n";
}