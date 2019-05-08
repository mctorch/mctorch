#include <torch/csrc/jit/passes/erase_number_types.h>
#include <torch/csrc/jit/constants.h>

namespace torch {
namespace jit {

static void EraseNumberTypesOnBlock(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto inp : it->inputs()) {
      if (inp->type()->isSubtypeOf(NumberType::get())) {
        inp->setType(TensorType::get());
      }
    }
    for (auto sub : it->blocks()) {
      EraseNumberTypesOnBlock(sub);
    }
    switch (it->kind()) {
      case prim::Constant: {
        // remove primitive constants, replacing with tensor equivalent
        // ONNX does not support non-tensor constants
        if (it->output()->type()->isSubtypeOf(NumberType::get()) ||
            it->output()->type()->isSubtypeOf(BoolType::get())) {
          at::Scalar s;
          if (it->output()->type()->isSubtypeOf(BoolType::get())) {
            s = static_cast<int64_t>(*constant_as<bool>(it->output()));
          } else {
            s = *constant_as<at::Scalar>(it->output());
          }

          WithInsertPoint guard(*it);
          Value* r = block->owningGraph()->insertConstant(
              scalar_to_tensor(s), nullptr, c10::nullopt, it->scope());
          it->output()->replaceAllUsesWith(r);
        }
      } break;
      case prim::Bool:
      case prim::Float:
      case prim::Int:
      case prim::ImplicitTensorToNum:
      case prim::NumToTensor: {
        it->output()->replaceAllUsesWith(it->inputs()[0]);
        // Let DCE cleanup
      } break;
      default: {
        for (auto o : it->outputs()) {
          if (o->type()->isSubtypeOf(NumberType::get())) {
            o->setType(CompleteTensorType::fromNumberType(o->type()));
          } else if (o->type()->isSubtypeOf(BoolType::get())) {
            o->setType(CompleteTensorType::fromBoolType());
          }
        }
      } break;
    }
  }
}

void EraseNumberTypes(const std::shared_ptr<Graph>& graph) {
  EraseNumberTypesOnBlock(graph->block());
}
} // namespace jit
} // namespace torch
