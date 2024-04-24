#include <torch/extension.h>
#include <iostream>
#include <omp.h>
#include <tuple>
#include <time.h>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <execution>
#include <numeric>
#include <stdlib.h>

using namespace at;
using namespace torch;


torch::Tensor normal_multi_thread(float std, int n_emb, int dim, int n_cores){
  int unit = n_emb / n_cores;
  int remain = n_emb % n_cores;

  // allocate a memory space for output tensor
  torch::Tensor output = torch::empty({n_emb, dim}, torch::kFloat);
  auto output_a = output.accessor<float, 2>();

  #pragma omp parallel for num_threads(n_cores)
  for(int i = 0; i < n_cores; i++){
    torch::Generator generator = make_generator<CPUGeneratorImpl>();
    generator.set_current_seed(rand());
    if(i == n_cores - 1 && remain != 0){
      torch::Tensor output_slice = output.index({torch::indexing::Slice(unit*i, unit*(i+1) + remain)});
      torch::normal_out(output_slice, 0, std, {unit + remain, dim}, generator);
    }
    else{
      torch::Tensor output_slice = output.index({torch::indexing::Slice(unit*i, unit*(i+1))});
      torch::normal_out(output_slice, 0, std, {unit, dim}, generator);
    }
  }
  return output;
}


torch::Tensor normal_multi_thread_with_extra(const torch::Tensor &std, int dim, int extra, int n_cores){
  int n_emb = std.sizes()[0]; // dimension of std: (n_emb)
  int unit = n_emb / n_cores;
  int remain = n_emb % n_cores;

  // allocate a memory space for output tensor
  torch::Tensor output = torch::empty({n_emb + extra, dim});

  #pragma omp parallel for num_threads(n_cores)
  for(int i = 0; i < n_cores; i++){
    torch::Generator generator = make_generator<CPUGeneratorImpl>();
    generator.set_current_seed(rand());
    if(i == n_cores - 1 && remain != 0){
      torch::Tensor output_slice = output.index({torch::indexing::Slice(unit*i, unit*(i+1) + remain)});
      torch::normal_out(output_slice, 0, 1, {unit + remain, dim}, generator);
    }
    else{
      torch::Tensor output_slice = output.index({torch::indexing::Slice(unit*i, unit*(i+1))});
      torch::normal_out(output_slice, 0, 1, {unit, dim}, generator);
    }
  }
  output.index({torch::indexing::Slice(0, n_emb)}).mul_(std.view({-1, 1}));
  return output;
}


torch::Tensor unique_multi_thread(const torch::Tensor &input){
  std::vector<long int> input_vector(input.data<long int>(), input.data<long int>() + input.numel());
  
  std::sort(std::execution::par_unseq, input_vector.begin(), input_vector.end());
  
  auto last = std::unique(std::execution::par_unseq, input_vector.begin(), input_vector.end());
  input_vector.erase(last, input_vector.end());
  torch::Tensor output = torch::empty({input_vector.size()}, torch::kInt64);
  memcpy(output.data<long int>(), &input_vector[0], input_vector.size() * sizeof(long int));

  return output;
}


typedef std::pair<long int, long int> int_pair;

torch::Tensor coalesce_multi_thread_openmp(const torch::Tensor &input, int n_cores){
  // If input tensor is already coalesced, just return
  if(input.is_coalesced()){
    return input;
  }

  // Convert data types of indices and values from torch::Tensor to std::vector<long int>
  torch::Tensor indices = input._indices();
  torch::Tensor values = input._values();

  // Set several variables
  int n_embs = input.sizes()[0];
  int n_rows = values.sizes()[0];
  int dim = values.sizes()[1];
  assert(dim == input.sizes()[1]);
  assert(indices.sizes()[0] == 1);
  assert(indices.sizes()[1] == n_rows);

  // Create a vector of pairs (indices, new indices started from 0)
  std::vector<long int> indices_vector(indices.data<long int>(), indices.data<long int>() + indices.numel());
  std::vector<std::pair<long int, long int>> indices_vector_with_index(n_rows);
  for(int i = 0; i < n_rows; i++){
    indices_vector_with_index[i].first = indices_vector[i];
    indices_vector_with_index[i].second = i;
  }

  // Sort that vector of pairs
  std::sort(std::execution::par_unseq, indices_vector_with_index.begin(), indices_vector_with_index.end(), [](const std::pair<long int, long int> lhs, const std::pair<long int, long int> rhs){
    return lhs.first < rhs.first;
  });
  
  // Do coalescing and derive start, end indices for each coalesced index
  std::vector<long int> coalesced_indices_vector;
  std::vector<long int> start_indices;
  std::vector<long int> end_indices;
  for(int i = 0; i < n_rows; i++){
    if(i == 0 || indices_vector_with_index[i].first != indices_vector_with_index[i-1].first){
      if(i != 0){
        end_indices.push_back(i - 1);
      }
      coalesced_indices_vector.push_back(indices_vector_with_index[i].first);
      start_indices.push_back(i);
    }
  }
  end_indices.push_back(n_rows - 1);
  int n_coalesced_rows = coalesced_indices_vector.size();


  // Derive coalesced values for each coalesced index
  torch::Tensor out_values = torch::empty({n_coalesced_rows, dim}, torch::kFloat);
  assert(out_values.is_contiguous());
  assert(values.is_contiguous());

  #pragma omp parallel for num_threads(n_cores)
  for(int i = 0; i < n_coalesced_rows; i++){
    memcpy(out_values.data<float>() + i * dim, values.data<float>() + indices_vector_with_index[start_indices[i]].second * dim, dim * sizeof(float));
    if(end_indices[i] > start_indices[i]){
      for(int j = start_indices[i] + 1; j <= end_indices[i]; j++){
        out_values.index({i}).add_(values.index({indices_vector_with_index[j].second}));
      }
    }
  }

  // 5
  torch::Tensor out_indices = torch::empty({1, coalesced_indices_vector.size()}, torch::kInt64);
  memcpy(out_indices.data<long int>(), &coalesced_indices_vector[0], coalesced_indices_vector.size() * sizeof(long int));
  torch::Tensor output = torch::sparse_coo_tensor(out_indices, out_values, {n_embs, dim});
  output._coalesced_(true);
  
  return output;
}


torch::Tensor coalesce_multi_thread_embeddingbag(const torch::Tensor &input, int n_cores){
  // If input tensor is already coalesced, just return
  if(input.is_coalesced()){
    return input;
  }

  // Convert data types of indices and values from torch::Tensor to std::vector<long int>
  torch::Tensor indices = input._indices();
  torch::Tensor values = input._values();

  // Set several variables
  int n_embs = input.sizes()[0];
  int n_rows = values.sizes()[0];
  int dim = values.sizes()[1];
  assert(dim == input.sizes()[1]);
  assert(indices.sizes()[0] == 1);
  assert(indices.sizes()[1] == n_rows);

  // 1. Create a vector of pairs (indices, new indices started from 0)
  std::vector<long int> indices_vector(indices.data<long int>(), indices.data<long int>() + indices.numel());
  std::vector<int_pair> indices_vector_with_index(n_rows);
  std::for_each(std::execution::par_unseq, indices_vector_with_index.begin(), indices_vector_with_index.end(), [&](int_pair &pair){
    unsigned long int i = (uintptr_t(&pair) - uintptr_t(indices_vector_with_index.data())) / sizeof(int_pair); 
    pair.first = indices_vector[i];
    pair.second = i;
  });

  // 2. Sort that vector of pairs
  std::sort(std::execution::par_unseq, indices_vector_with_index.begin(), indices_vector_with_index.end(), [](const std::pair<long int, long int> lhs, const std::pair<long int, long int> rhs){
    return lhs.first < rhs.first;
  });
  
  // 3. Extract each elements and form seperated vector
  std::vector<long int> sorted_first(indices.numel());
  std::vector<long int> embedding_idx(indices.numel());
  std::for_each(std::execution::par_unseq, indices_vector_with_index.begin(), indices_vector_with_index.end(), [&](const int_pair &pair){
    unsigned long int i = (uintptr_t(&pair) - uintptr_t(indices_vector_with_index.data())) / sizeof(int_pair); 
    sorted_first[i] = pair.first;
    embedding_idx[i] = pair.second;
  });
  
  // 4. Derive coalesced_indices by applying unique() to sorted_first
  auto last = std::unique(std::execution::par_unseq, sorted_first.begin(), sorted_first.end());
  sorted_first.erase(last, sorted_first.end());
  torch::Tensor coalesced_indices = torch::empty({1, sorted_first.size()}, torch::kInt64);
  memcpy(coalesced_indices.data<long int>(), &sorted_first[0], sorted_first.size() * sizeof(long int));
  
  // 5. Derive embedding_offsets
  std::vector<long int> embedding_offset(sorted_first.size());
  std::vector<long int> difference_occur(n_rows);
  std::vector<long int> result_exclusive_scan(n_rows);
  std::for_each(std::execution::par_unseq, difference_occur.begin(), difference_occur.end(), [&](long int &e){
    unsigned long int i = (uintptr_t(&e) - uintptr_t(difference_occur.data())) / sizeof(long int); 
    if((i == 0) || (indices_vector_with_index[i-1].first != indices_vector_with_index[i].first)){
      e = 1;
    }else{
      e = 0;
    }
  });
  std::exclusive_scan(std::execution::par_unseq, difference_occur.begin(), difference_occur.end(), result_exclusive_scan.begin(), 0);
  std::for_each(std::execution::par_unseq, difference_occur.begin(), difference_occur.end(), [&](long int &e){
    if(e == 1){
      unsigned long int i = (uintptr_t(&e) - uintptr_t(difference_occur.data())) / sizeof(long int); 
      embedding_offset[result_exclusive_scan[i]] = i;
    }
  });

  // 6. Convert vector to torch.tensor for embedding_bag
  torch::Tensor embedding_idx_tensor = torch::empty({embedding_idx.size()}, torch::kInt64);
  memcpy(embedding_idx_tensor.data<long int>(), &embedding_idx[0], embedding_idx.size() * sizeof(long int));
  torch::Tensor embedding_offset_tensor = torch::empty({embedding_offset.size()}, torch::kInt64);
  memcpy(embedding_offset_tensor.data<long int>(), &embedding_offset[0], embedding_offset.size() * sizeof(long int));

  // 7. Apply embedding_bag
  torch::Tensor coalesced_values = std::get<0>(torch::_embedding_bag_forward_only(values, embedding_idx_tensor, embedding_offset_tensor));
  
  // 8. Form the answer as a sparse_coo_tensor
  torch::Tensor output = torch::sparse_coo_tensor(coalesced_indices, coalesced_values, {n_embs, dim});
  output._coalesced_(true);
  return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("normal_multi_thread", &normal_multi_thread, "This function samples the random variables that follow Gaussian distribution. It only supports the case whose mean is 0 and the standard devication is a fixed value. The output of this function is a 2D tensor whose shape is \"n_emb\"x\"dim\" and whose entries follow gaussain random variable of mean 0 and standard deviation \"std\".");
  m.def("normal_multi_thread_with_extra", &normal_multi_thread_with_extra, "This function samples the random variables that follow Gaussian distribution. It allocates the larger memory space (the \"extra\") to store the gradients derived in backward propagation. Also, this function gets a 1D tensor, \"std\" as a input to generate Gaussian random variables with different stadard derivation in a row granularity");
  m.def("unique_multi_thread", &unique_multi_thread, "This funciton does an exact same thing with torch.unique(), but using multiple threads.");
  m.def("coalesce_multi_thread_openmp", &coalesce_multi_thread_openmp, "This funciton does an exact same things with torch.coalesce(), but using multiple threads. This function is implemented by C++ stadard library and OpenMP");
  m.def("coalesce_multi_thread_embeddingbag", &coalesce_multi_thread_embeddingbag, "This funciton does an exact same things with torch.coalesce(), but using multiple threads. This function is implemented by C++ stadard library and \"torch::_embedding_bag_forward_only\"");
}
