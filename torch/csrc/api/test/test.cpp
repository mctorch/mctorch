#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <torch/torch.h>

#include <iostream>

TEST_CASE("C++", "[cpp]") {
  // This compiles.
  torch::nn::LSTM lstm(4, 8);

  torch::nn::Sequential model(
      torch::nn::LSTM(2, 2),
      torch::nn::LSTM(2, 2),
      torch::nn::LSTM(2, 2),
      torch::nn::LSTM(2, 2));

  model.append(torch::nn::LSTM(2, 2));

  // model.modules().apply([](const std::string& key, Module& module) {
  //   std::cout << key << ": " << module.name() << std::endl;
  // });
}
