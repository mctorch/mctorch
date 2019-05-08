#include <gtest/gtest.h>

#include <torch/jit.h>
#include <torch/types.h>

#include <string>

TEST(TorchScriptTest, CanCompileMultipleFunctions) {
  auto module = torch::jit::compile(R"JIT(
      def test_mul(a, b):
        return a * b
      def test_relu(a, b):
        return torch.relu(a + b)
      def test_while(a, i):
        while bool(i < 10):
          a += a
          i += 1
        return a
      def test_len(a : List[int]):
        return len(a)
    )JIT");
  auto a = torch::ones(1);
  auto b = torch::ones(1);

  ASSERT_EQ(1, module->run_method("test_mul", a, b).toTensor().item<int64_t>());

  ASSERT_EQ(2, module->run_method("test_relu", a, b).toTensor().item<int64_t>());

  ASSERT_TRUE(
      0x200 == module->run_method("test_while", a, b).toTensor().item<int64_t>());

  at::IValue list = std::vector<int64_t>({3, 4});
  ASSERT_EQ(2, module->run_method("test_len", list).toInt());

}


TEST(TorchScriptTest, TestNestedIValueModuleArgMatching) {
  auto module = torch::jit::compile(R"JIT(
      def nested_loop(a: List[List[Tensor]], b: int):
        return torch.tensor(1.0) + b
    )JIT");

  auto b = 3;

  std::vector<torch::Tensor> list = {torch::rand({4, 4})};

  std::vector<torch::jit::IValue> list_of_lists;
  list_of_lists.push_back(list);
  module->run_method("nested_loop", list_of_lists, b);

  std::vector<torch::jit::IValue> generic_list;
  std::vector<torch::jit::IValue> empty_generic_list;
  empty_generic_list.push_back(generic_list);
  module->run_method("nested_loop", empty_generic_list, b);

  std::vector<torch::jit::IValue> too_many_lists;
  too_many_lists.push_back(empty_generic_list);
  try {
    module->run_method("nested_loop", too_many_lists, b);
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("Expected value of type Tensor[][] for argument 'a' in "
                  "position 0, but instead got value of type t[][][]") == 0);

  };

  std::vector<torch::jit::IValue> gen_list;
  std::vector<int64_t> int_list = {1, 2, 3};

  gen_list.emplace_back(list);
  gen_list.emplace_back(int_list);

  try {
    module->run_method("nested_loop", gen_list, b);
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    //TODO: currently does not unify types across encounted generic lists,
    //so the error message is not helpful here.
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("Expected value of type Tensor[][] for argument 'a' in "
                  "position 0, but instead got value of type Tensor[][]") == 0);

  };
}


TEST(TorchScriptTest, TestDictArgMatching) {
  auto module = torch::jit::compile(R"JIT(
      def dict_op(a: Dict[str, Tensor], b: str):
        return a[b]
    )JIT");
  c10::ivalue::UnorderedMap dict;
  dict[std::string("hello")] = torch::ones({2});
  auto output = module->run_method("dict_op", dict, std::string("hello"));
  ASSERT_EQ(1, output.toTensor()[0].item<int64_t>());
}

TEST(TorchScriptTest, TestTupleArgMatching) {
  auto module = torch::jit::compile(R"JIT(
      def tuple_op(a: Tuple[List[int]]):
        return a
    )JIT");

  std::vector<int64_t> int_list = {1};
  auto tuple_generic_list = torch::jit::Tuple::create({ int_list });

  // doesn't fail on arg matching
  module->run_method("tuple_op", tuple_generic_list);

}

TEST(TorchScriptTest, TestOptionalArgMatching) {
  auto module = torch::jit::compile(R"JIT(
      def optional_tuple_op(a: Optional[Tuple[int, str]]):
        if a is None:
          return 0
        else:
          return a[0]
    )JIT");

  auto optional_tuple = torch::jit::Tuple::create({2, std::string("hi")});

  ASSERT_EQ(2, module->run_method("optional_tuple_op", optional_tuple).toInt());
  ASSERT_EQ(
      0, module->run_method("optional_tuple_op", torch::jit::IValue()).toInt());

}
