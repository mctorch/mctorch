#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/testing/file_check.h>
#include "test/cpp/jit/test_base.h"

#include <sstream>
#include <string>

namespace torch {
namespace jit {

void testConstantPooling() {
  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
graph():
  %8 : int = prim::Constant[value=1]()
  %10 : int = prim::Constant[value=1]()
  return (%8, %10)
  )IR",
        &*graph);
    ConstantPooling(graph);
    testing::FileCheck()
        .check_count("prim::Constant", 1, /*exactly*/ true)
        ->run(*graph);
  }
  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
graph(%cond : Tensor):
  %a : str = prim::Constant[value="bcd"]()
  %3 : bool = prim::Bool(%cond)
  %b : str = prim::If(%3)
    block0():
      %b.1 : str = prim::Constant[value="abc"]()
      -> (%b.1)
    block1():
      %b.2 : str = prim::Constant[value="abc"]()
      -> (%b.2)
  %7 : (str, str) = prim::TupleConstruct(%a, %b)
  return (%7)
  )IR",
        &*graph);
    ConstantPooling(graph);
    testing::FileCheck()
        .check_count("prim::Constant[value=\"abc\"]", 1, /*exactly*/ true)
        ->check_count("prim::Constant[value=\"bcd\"]", 1, /*exactly*/ true)
        ->run(*graph);
  }
  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
graph():
  %2 : int = prim::Constant[value=2]()
  %1 : int = prim::Constant[value=1]()
  %5 : int? = prim::Constant()
  %7 : Device? = prim::Constant()
  %15: bool = prim::Constant[value=0]()
  %10 : int = prim::Constant[value=6]()
  %3 : int[] = prim::ListConstruct(%1, %2)
  %x : Tensor = aten::tensor(%3, %5, %7, %15)
  %y : Tensor = aten::tensor(%3, %10, %7, %15)
  %9 : int[] = prim::ListConstruct(%1, %2)
  %z : Tensor = aten::tensor(%9, %10, %7, %15)
  %f = prim::Print(%x, %y, %z)
  return (%1)
  )IR",
        &*graph);
    // three tensors created - two different devices among the three
    // don't have good support for parsing tensor constants
    ConstantPropagation(graph);
    ConstantPooling(graph);
    testing::FileCheck()
        .check_count("Float(2) = prim::Constant", 1, /*exactly*/ true)
        ->check_count("Long(2) = prim::Constant", 1, /*exactly*/ true)
        ->run(*graph);
  }
  // don't create aliasing of graph outputs in constant pooling
  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
graph(%cond : Tensor):
  %a : Tensor = prim::Constant()
  %b : Tensor = prim::Constant()
  %c : Tensor = prim::Constant()
  %1 = prim::Print(%c)
  return (%a, %b)
  )IR",
        &*graph);
    ConstantPooling(graph);
    testing::FileCheck()
        .check_count("prim::Constant", 2, /*exactly*/ true)
        ->run(*graph);
  }
}
} // namespace jit
} // namespace torch
