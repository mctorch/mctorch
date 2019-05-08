#include <ATen/native/RNN.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at { namespace native {

namespace {

template<typename T>
using pair_of = std::pair<T, T>;

template<typename T>
using tpair_of = std::tuple<T, T>;

// Those could have been function pointers, but MSVC chokes on function pointers as template parameters
struct tanh_f {
  Tensor operator()(const Tensor& t) const { return at::tanh(t); }
};

struct relu_f {
  Tensor operator()(const Tensor& t) const { return at::relu(t); }
};

struct PackedSequence {
  PackedSequence() = default;
  PackedSequence(Tensor _data, Tensor _batch_sizes)
    : data(std::move(_data)), batch_sizes(std::move(_batch_sizes)) {}

  Tensor data;
  Tensor batch_sizes;
};

// Pretty much all cells we support take the same set of arguments, but threading those
// 4 arguments manually is really annoying. Their lifetime is externally managed, so we only
// pass this struct of references around.
struct CellParams {
  CellParams(const Tensor& _w_ih, const Tensor& _w_hh, const Tensor& _b_ih, const Tensor& _b_hh)
    : w_ih(_w_ih), w_hh(_w_hh), b_ih(_b_ih), b_hh(_b_hh) {};

  const Tensor& w_ih;
  const Tensor& w_hh;
  const Tensor& b_ih; /* optional */
  const Tensor& b_hh; /* optional */

  Tensor matmul_ih(Tensor input) const {
    return at::matmul(input, w_ih.t());
  }
  Tensor matmul_hh(Tensor h) const {
    return at::matmul(h, w_hh.t());
  }
  Tensor linear_ih(Tensor input) const {
    return at::linear(input, w_ih, b_ih);
  }
  Tensor linear_hh(Tensor h) const {
    return at::linear(h, w_hh, b_hh);
  }
};

// Run this Python script and pipe to clang-format to generate the constructor
// and data members:
//
// names = ['w', 'b', 'packed', 'col_offsets', 'scale', 'zero_point']
//
//
// get_type = lambda i: 'Scalar' if i == 4 or i == 5 else 'Tensor'
// member_ref = lambda i: '' if i == 4 or i == 5 else '&'
//
// suffixes = ['ih', 'hh']
//
// params = []
// initializers = []
// members = []
// for i in range(len(names)*2):
//     params.append('const {typ}& _{name}_{suffix}'.format(typ=get_type(
//         i//2), name=names[(i//2) % len(names)], suffix=suffixes[i % 2]))
//     initializers.append('{name}_{suffix}(_{name}_{suffix})'.format(
//         name=names[(i//2) % len(names)], suffix=suffixes[i % 2]))
//     members.append('const {typ}{member_ref} {name}_{suffix};'.format(typ=get_type(
//         i//2), name=names[(i//2) % len(names)], suffix=suffixes[i % 2], member_ref=member_ref(i//2)))
//
// params_str = ', '.join(params)
// initializers_str = ', '.join(initializers)
// members_str = '\n'.join(members)
//
// ctor = 'QuantizedCellParams(' + params_str + ') : ' + initializers_str + '{}'
// print('struct QuantizedCellParams {', '\n\n'.join([ctor, members_str]), '};')

struct QuantizedCellParams {
  QuantizedCellParams(const Tensor &_w_ih, const Tensor &_w_hh,
                      const Tensor &_b_ih, const Tensor &_b_hh,
                      const Tensor &_packed_ih, const Tensor &_packed_hh,
                      const Tensor &_col_offsets_ih,
                      const Tensor &_col_offsets_hh, const Scalar &_scale_ih,
                      const Scalar &_scale_hh, const Scalar &_zero_point_ih,
                      const Scalar &_zero_point_hh)
      : w_ih(_w_ih), w_hh(_w_hh), b_ih(_b_ih), b_hh(_b_hh),
        packed_ih(_packed_ih), packed_hh(_packed_hh),
        col_offsets_ih(_col_offsets_ih), col_offsets_hh(_col_offsets_hh),
        scale_ih(_scale_ih), scale_hh(_scale_hh), zero_point_ih(_zero_point_ih),
        zero_point_hh(_zero_point_hh) {}

  const Tensor &w_ih;
  const Tensor &w_hh;
  const Tensor &b_ih;
  const Tensor &b_hh;
  const Tensor &packed_ih;
  const Tensor &packed_hh;
  const Tensor &col_offsets_ih;
  const Tensor &col_offsets_hh;
  const Scalar scale_ih;
  const Scalar scale_hh;
  const Scalar zero_point_ih;
  const Scalar zero_point_hh;

  Tensor matmul_ih(Tensor input) const {
    AT_CHECK(false, "matmul is not supported with quantized cell params");
  }
  Tensor matmul_hh(Tensor h) const {
    AT_CHECK(false, "matmul is not supported with quantized cell params");
  }
  Tensor linear_ih(Tensor input) const {
    return at::fbgemm_linear_int8_weight(
        input, w_ih, packed_ih, col_offsets_ih, scale_ih, zero_point_ih, b_ih);
  }
  Tensor linear_hh(Tensor h) const {
    return at::fbgemm_linear_int8_weight(
        h, w_hh, packed_hh, col_offsets_hh, scale_hh, zero_point_hh, b_hh);
  }
};

// Gathers every two elements of a vector in a vector of pairs
template<typename T>
static std::vector<pair_of<T>> pair_vec(const std::vector<T>& vals) {
  AT_CHECK(vals.size() % 2 == 0, "Odd number of params or hiddens given to a bidirectional RNN");
  std::vector<pair_of<T>> result;
  result.reserve(vals.size() / 2);
  for (int64_t i = 0; i < vals.size(); i += 2) {
    result.emplace_back(vals[i], vals[i + 1]);
  }
  return result;
}

// Flattens a vector of pairs
template<typename T>
static std::vector<T> unpair_vec(std::vector<pair_of<T>>&& vals) {
  std::vector<T> result;
  result.reserve(vals.size() * 2);
  for (int64_t i = 0; i < vals.size(); i++) {
    result.push_back(std::move(vals[i].first));
    result.push_back(std::move(vals[i].second));
  }
  return result;
}

// Parses a flat list of parameter tensors into a list of CellParams
static std::vector<CellParams> gather_params(TensorList params, bool has_biases) {
  static at::Tensor undefined;
  std::vector<CellParams> result;
  if (has_biases) {
    AT_CHECK(params.size() % 4 == 0, "got an incorrect number of RNN parameters");
    for (size_t i = 0; i < params.size(); i += 4) {
      result.emplace_back(params[i], params[i + 1], params[i + 2], params[i + 3]);
    }
  } else {
    AT_CHECK(params.size() % 2 == 0, "got an incorrect number of RNN parameters");
    for (size_t i = 0; i < params.size(); i += 2) {
      result.emplace_back(params[i], params[i + 1], undefined, undefined);
    }
  }
  return result;
}

static std::vector<QuantizedCellParams> gather_quantized_params(TensorList params) {
  static at::Tensor undefined;
  std::vector<QuantizedCellParams> result;
  AT_CHECK(params.size() % 12 == 0, "got an incorrect number of quantized RNN parameters");
  for (size_t i = 0; i < params.size(); i += 12) {
    result.emplace_back(params[i], params[i + 1], params[i + 2], params[i + 3],
                        params[i + 4], params[i + 5], params[i + 6], params[i + 7],
                        params[i + 8].item(), params[i + 9].item(),
                        params[i + 10].item(), params[i + 11].item());
  }
  return result;
}


////////////////////////////////////////////////////////////////////////////////
// HIDDEN STATE FUNCTIONS
//
// Functions implemented below are implemented as templates based on hidden type,
// because they need to work both with simple RNNs and GRU (which use a single Tensor),
// as well as with LSTM (or possibly more complicated architectures in the future).
// Still, there are some operations that need to be performed on the hidden states
// alone, and for this purpose we provide an overloaded set of functions below.

Tensor hidden_as_output(const Tensor& t) { return t; }
Tensor hidden_as_output(const tpair_of<Tensor>& t) { return std::get<0>(t); }

template<size_t index>
std::vector<Tensor> project(at::ArrayRef<tpair_of<Tensor>> tuples) {
  std::vector<Tensor> result;
  result.reserve(tuples.size());
  for (auto & t : tuples) {
    result.push_back(std::get<index>(t));
  }
  return result;
}

Tensor hidden_concat(at::ArrayRef<Tensor> hiddens) { return at::cat(hiddens, 0); }
tpair_of<Tensor> hidden_concat(at::ArrayRef<tpair_of<Tensor>> hiddens) {
  return std::make_tuple(hidden_concat(project<0>(hiddens)), hidden_concat(project<1>(hiddens)));
}

Tensor hidden_slice(const Tensor& t, int64_t start, int64_t end) {
  return t.narrow(0, start, end - start);
}
tpair_of<Tensor> hidden_slice(const tpair_of<Tensor>& t, int64_t start, int64_t end) {
  return std::make_tuple(hidden_slice(std::get<0>(t), start, end),
                         hidden_slice(std::get<1>(t), start, end));
}

////////////////////////////////////////////////////////////////////////////////
// CELL IMPLEMENTATIONS
//
// Cell is a basic component of an RNN, representing a single application of the
// recurrent function. You can think of it as a function of signature
//
// (Tensor input, hidden_type hidden, CellParams) -> hidden_type
//
// which means that it consumes an input tensor, and updates the previous hidden state.
// It's a struct only because functional programming in C++ is a pain, and it's easier
// to pass around "vtable pointers" than actual function pointers.

template<typename hidden_type_tmpl, typename cell_params_tmpl>
struct Cell {
  using hidden_type = hidden_type_tmpl;
  using cell_params = cell_params_tmpl;
  virtual ~Cell() {} // This is really dumb, but enables projects with -Wnon-virtual-dtor to compile...
  virtual hidden_type operator()(const Tensor& input, const hidden_type& hidden, const cell_params& params) const = 0;
};

template<typename nonlinearity, typename cell_params>
struct SimpleCell : Cell<Tensor, cell_params> {
  using hidden_type = Tensor;
  Tensor operator()(const Tensor& input, const Tensor& hidden, const cell_params& params) const override {
    return nonlinearity{}(params.linear_ih(input) + params.linear_hh(hidden));
  }
};

// TODO: can use inplace ops?
template <typename cell_params>
struct LSTMCell : Cell<std::tuple<Tensor, Tensor>, cell_params> {
  using hidden_type = std::tuple<Tensor, Tensor>;
  hidden_type operator()(const Tensor& input, const hidden_type& hidden, const cell_params& params) const override {
    auto hx = std::get<0>(hidden);
    auto cx = std::get<1>(hidden);

    if (input.is_cuda()) {
      auto igates = params.matmul_ih(input);
      auto hgates = params.matmul_hh(hx);
      auto result = at::_thnn_fused_lstm_cell(igates, hgates, cx, params.b_ih, params.b_hh);
      // Slice off the workspace argument (it's needed only for AD).
      return std::make_tuple(std::get<0>(result), std::get<1>(result));
    }

    auto gates = params.linear_ih(input) + params.linear_hh(hx);
    auto chunked_gates = gates.chunk(4, 1);

    auto ingate = chunked_gates[0].sigmoid();
    auto forgetgate = chunked_gates[1].sigmoid();
    auto cellgate = chunked_gates[2].tanh();
    auto outgate = chunked_gates[3].sigmoid();

    auto cy = (forgetgate * cx) + (ingate * cellgate);
    auto hy = outgate * cy.tanh();

    return std::make_tuple(hy, cy);
  }
};

template <typename cell_params>
struct GRUCell : Cell<Tensor, cell_params> {
  using hidden_type = Tensor;
  hidden_type operator()(const Tensor& input, const hidden_type& hidden, const cell_params& params) const override {
    if (input.is_cuda()) {
      auto igates = params.matmul_ih(input);
      auto hgates = params.matmul_hh(hidden);
      auto result = at::_thnn_fused_gru_cell(igates, hgates, hidden, params.b_ih, params.b_hh);
      // Slice off the workspace argument (it's needed only for AD).
      return std::get<0>(result);
    }

    auto igates = params.linear_ih(input);
    auto hgates = params.linear_hh(hidden);
    auto chunked_igates = igates.chunk(3, 1);
    auto chunked_hgates = hgates.chunk(3, 1);

    auto reset_gate = at::sigmoid(chunked_igates[0] + chunked_hgates[0]);
    auto input_gate = at::sigmoid(chunked_igates[1] + chunked_hgates[1]);
    auto new_gate = at::tanh(chunked_igates[2] + reset_gate * chunked_hgates[2]);

    return new_gate + input_gate * (hidden - new_gate);
  }
};

////////////////////////////////////////////////////////////////////////////////
// LAYER IMPLEMENTATIONS
//
// Layers are scan-like higher-order functions, which take in cells, and
// transform them to functions of signature
//
// (io_type input, hidden_type hidden, param_type params) -> (io_type, hidden_type)
//
// which can apply the cell over a sequence of inputs, and produce both a new set
// of hidden states, as well as a concatenated output of each step.

template<typename output_type, typename hidden_type>
struct LayerOutput {
  output_type outputs;
  hidden_type final_hidden;
};

template<typename io_type, typename hidden_type, typename param_type>
struct Layer {
  using output_type = LayerOutput<io_type, hidden_type>;
  virtual ~Layer() {} // This is really dumb, but enables projects with -Wnon-virtual-dtor to compile...
  virtual output_type operator()(const io_type& input, const hidden_type& input_hidden, const param_type& params) const = 0;
};

template<typename hidden_type, typename cell_params>
struct FullLayer : Layer<Tensor, hidden_type, cell_params> {
  using output_type = typename Layer<Tensor, hidden_type, cell_params>::output_type;
  using unstacked_output_type = LayerOutput<std::vector<Tensor>, hidden_type>;

  FullLayer(Cell<hidden_type, cell_params>& cell)
    : cell_(cell) {};

  unstacked_output_type operator()(std::vector<Tensor> step_inputs, const hidden_type& input_hidden, const cell_params& params) const {
    std::vector<Tensor> step_outputs;
    auto hidden = input_hidden;
    for (size_t i = 0; i < step_inputs.size(); i++) {
      hidden = cell_(step_inputs[i], hidden, params);
      step_outputs.push_back(hidden_as_output(hidden));
    }
    return {step_outputs, hidden};
  }

  output_type operator()(const Tensor& inputs, const hidden_type& input_hidden, const cell_params& params) const override {
    auto unstacked_output = (*this)(inputs.unbind(0), input_hidden, params);
    return {at::stack(unstacked_output.outputs, 0), unstacked_output.final_hidden};
  }

  Cell<hidden_type, cell_params>& cell_;
};

template<typename dir_hidden_type, typename cell_params>
struct FullBidirectionalLayer : Layer<Tensor, pair_of<dir_hidden_type>, pair_of<cell_params>> {
  using hidden_type = pair_of<dir_hidden_type>;
  using param_type = pair_of<cell_params>;
  using output_type = typename Layer<Tensor, hidden_type, param_type>::output_type;

  FullBidirectionalLayer(Cell<dir_hidden_type, cell_params>& cell)
    : layer_(cell) {};

  output_type operator()(const Tensor& input, const hidden_type& input_hidden, const param_type& params) const override {
    auto step_inputs = input.unbind(0);
    auto fw_result = layer_(step_inputs, input_hidden.first, params.first);
    auto fw_output = at::stack(fw_result.outputs, 0);

    auto rev_step_inputs = reverse(std::move(step_inputs));
    auto rev_result = layer_(rev_step_inputs, input_hidden.second, params.second);
    std::reverse(rev_result.outputs.begin(), rev_result.outputs.end());
    auto rev_output = at::stack(rev_result.outputs, 0);

    return {at::cat({fw_output, rev_output}, fw_output.dim() - 1),
            std::make_pair(fw_result.final_hidden, rev_result.final_hidden)};
  }

  std::vector<Tensor> reverse(std::vector<Tensor>&& x) const {
    std::reverse(x.begin(), x.end());
    return std::move(x);
  }

  FullLayer<dir_hidden_type, cell_params> layer_;
};

template<typename hidden_type, typename cell_params>
struct PackedLayer : Layer<PackedSequence, hidden_type, cell_params> {
  using output_type = typename Layer<PackedSequence, hidden_type, cell_params>::output_type;

  PackedLayer(Cell<hidden_type, cell_params>& cell)
    : cell_(cell) {};

  output_type operator()(const PackedSequence& input, const hidden_type& input_hidden, const cell_params& params) const override {
    std::vector<at::Tensor> step_outputs;
    std::vector<hidden_type> hiddens;
    int64_t input_offset = 0;
    int64_t num_steps = input.batch_sizes.size(0);
    int64_t* batch_sizes = input.batch_sizes.data<int64_t>();
    int64_t last_batch_size = batch_sizes[0];

    // Batch sizes is a sequence of decreasing lengths, which are offsets
    // into a 1D list of inputs. At every step we slice out batch_size elements,
    // and possibly account for the decrease in the batch size since the last step,
    // which requires us to slice the hidden state (since some sequences
    // are completed now). The sliced parts are also saved, because we will need
    // to return a tensor of final hidden state.
    auto hidden = input_hidden;
    for (int64_t i = 0; i < num_steps; ++i) {
      int64_t batch_size = batch_sizes[i];
      auto step_input = input.data.narrow(0, input_offset, batch_size);
      input_offset += batch_size;

      int64_t dec = last_batch_size - batch_size;
      if (dec > 0) {
        hiddens.push_back(hidden_slice(hidden, last_batch_size - dec, last_batch_size));
        hidden = hidden_slice(hidden, 0, last_batch_size - dec);
      }

      last_batch_size = batch_size;
      hidden = cell_(step_input, hidden, params);
      step_outputs.push_back(hidden_as_output(hidden));
    }
    hiddens.push_back(hidden);
    std::reverse(hiddens.begin(), hiddens.end());

    return { PackedSequence{ at::cat(step_outputs, 0), input.batch_sizes }, hidden_concat(hiddens) };
  }

  Cell<hidden_type, cell_params>& cell_;
};

template<typename hidden_type, typename cell_params>
struct ReversedPackedLayer : Layer<PackedSequence, hidden_type, cell_params> {
  using output_type = typename Layer<PackedSequence, hidden_type, cell_params>::output_type;

  ReversedPackedLayer(Cell<hidden_type, cell_params>& cell)
    : cell_(cell) {};

  output_type operator()(const PackedSequence& input, const hidden_type& input_hidden, const cell_params& params) const override {
    std::vector<at::Tensor> step_outputs;
    int64_t input_offset = input.data.size(0);
    int64_t num_steps = input.batch_sizes.size(0);
    int64_t* batch_sizes = input.batch_sizes.data<int64_t>();
    int64_t last_batch_size = batch_sizes[num_steps - 1];

    // Here the situation is similar to that above, except we start out with
    // the smallest batch size (and a small set of hidden states we actually use),
    // and progressively expand the hidden states, as we move backwards over the
    // 1D list of inputs.
    auto hidden = hidden_slice(input_hidden, 0, batch_sizes[num_steps - 1]);
    for (int64_t i = num_steps - 1; i >= 0; --i) {
      int64_t batch_size = batch_sizes[i];
      int64_t inc = batch_size - last_batch_size;
      if (inc > 0) {
        hidden = hidden_concat(ArrayRef<hidden_type>{hidden, hidden_slice(input_hidden, last_batch_size, batch_size)});
      }

      auto step_input = input.data.narrow(0, input_offset - batch_size, batch_size);
      input_offset -= batch_size;

      last_batch_size = batch_size;
      hidden = cell_(step_input, hidden, params);
      step_outputs.push_back(hidden_as_output(hidden));
    }
    std::reverse(step_outputs.begin(), step_outputs.end());
    return { PackedSequence{ at::cat(step_outputs, 0), input.batch_sizes }, hidden };
  }

  Cell<hidden_type, cell_params>& cell_;
};

template<typename dir_hidden_type, typename cell_params>
struct PackedBidirectionalLayer : Layer<PackedSequence, pair_of<dir_hidden_type>, pair_of<cell_params>> {
  using hidden_type = pair_of<dir_hidden_type>;
  using param_type = pair_of<cell_params>;
  using output_type = typename Layer<PackedSequence, hidden_type, param_type>::output_type;

  PackedBidirectionalLayer(Cell<dir_hidden_type, cell_params>& cell)
    : layer_(cell), rev_layer_(cell) {};

  output_type operator()(const PackedSequence& input, const hidden_type& input_hidden, const param_type& params) const override {
    auto fw_result = layer_(input, input_hidden.first, params.first);
    auto rev_result = rev_layer_(input, input_hidden.second, params.second);
    PackedSequence output { at::cat({fw_result.outputs.data, rev_result.outputs.data}, -1), input.batch_sizes };
    return { output, std::make_pair(fw_result.final_hidden, rev_result.final_hidden) };
  }

  PackedLayer<dir_hidden_type, cell_params> layer_;
  ReversedPackedLayer<dir_hidden_type, cell_params> rev_layer_;
};

////////////////////////////////////////////////////////////////////////////////
// apply_layer_stack
//
// layers are convenient, but in reality we often want to stack them. this little
// helper manages slicing of all inputs and parameters, and repeatedly feeds them
// into the given layer. returns the last layer's outputs, and a vector of final
// hidden states produced at each level.

Tensor dropout(const Tensor& input, double p) {
  return at::dropout(input, p, /*train=*/true);
}

PackedSequence dropout(const PackedSequence& input, double p) {
  return {at::dropout(input.data, p, /*train=*/true), input.batch_sizes};
}

template<typename io_type, typename hidden_type, typename weight_type>
LayerOutput<io_type, std::vector<hidden_type>>
apply_layer_stack(const Layer<io_type, hidden_type, weight_type>& layer, const io_type& input,
                  const std::vector<hidden_type>& hiddens, const std::vector<weight_type>& weights,
                  int64_t num_layers, double dropout_p, bool train) {
  AT_CHECK(num_layers == hiddens.size(), "Expected more hidden states in stacked_rnn");
  AT_CHECK(num_layers == weights.size(), "Expected more weights in stacked_rnn");

  auto layer_input = input;
  auto hidden_it = hiddens.begin();
  auto weight_it = weights.begin();
  std::vector<hidden_type> final_hiddens;
  for (int64_t l = 0; l < num_layers; ++l) {
    auto layer_output = layer(layer_input, *(hidden_it++), *(weight_it++));
    final_hiddens.push_back(layer_output.final_hidden);
    layer_input = layer_output.outputs;

    if (dropout_p != 0 && train && l < num_layers - 1) {
      layer_input = dropout(layer_input, dropout_p);
    }
  }

  return {layer_input, final_hiddens};
}

////////////////////////////////////////////////////////////////////////////////
// HELPERS SIMPLIFYING DISPATCH TO FUNCTIONS ABOVE
////////////////////////////////////////////////////////////////////////////////

template<typename CellType, template<typename,typename> class LayerT, template<typename,typename> class BidirLayerT, typename cell_params, typename io_type>
LayerOutput<io_type, std::vector<typename CellType::hidden_type>> _rnn_impl(
      const io_type& input,
      const std::vector<cell_params>& params,
      const std::vector<typename CellType::hidden_type>& hiddens,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {
  using hidden_type = typename CellType::hidden_type;
  CellType cell;
  if (bidirectional) {
    using BidirLayer = BidirLayerT<hidden_type, cell_params>;
    auto bidir_result = apply_layer_stack(BidirLayer{cell}, input, pair_vec(hiddens), pair_vec(params), num_layers, dropout_p, train);
    return {bidir_result.outputs, unpair_vec(std::move(bidir_result.final_hidden))};
  } else {
    return apply_layer_stack(LayerT<hidden_type,cell_params>{cell}, input, hiddens, params, num_layers, dropout_p, train);
  }
}

template<typename CellType, template<typename,typename> class LayerT, template<typename,typename> class BidirLayerT, typename cell_params, typename io_type>
std::tuple<io_type, Tensor> _rnn_impl_with_concat(
      const io_type& input,
      const std::vector<cell_params>& params,
      const std::vector<typename CellType::hidden_type>& hiddens,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {
  auto result = _rnn_impl<CellType, LayerT, BidirLayerT>(input, params, hiddens, num_layers, dropout_p, train, bidirectional);
  return std::make_tuple(result.outputs, at::stack(result.final_hidden, 0));
}

template<template<typename,typename> class LayerT, template<typename,typename> class BidirLayerT, typename cell_params, typename io_type>
std::tuple<io_type, Tensor, Tensor> _lstm_impl(
      const io_type& input,
      const std::vector<cell_params>& params, const Tensor& hx, const Tensor& cx,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {
  // It's much more useful for us to work on lists of pairs of hx and cx for each layer, so we need
  // to transpose a pair of those tensors.
  auto layer_hx = hx.unbind(0);
  auto layer_cx = cx.unbind(0);
  int64_t total_layers = layer_hx.size();
  std::vector<typename LSTMCell<cell_params>::hidden_type> hiddens;
  hiddens.reserve(total_layers);
  for (int64_t i = 0; i < total_layers; ++i) {
    hiddens.emplace_back(std::move(layer_hx[i]), std::move(layer_cx[i]));
  }

  auto result = _rnn_impl<LSTMCell<cell_params>, LayerT, BidirLayerT>(input, params, hiddens, num_layers, dropout_p, train, bidirectional);

  // Now, we need to reverse the transposed we performed above.
  std::vector<Tensor> hy, cy;
  hy.reserve(total_layers); cy.reserve(total_layers);
  for (auto & hidden : result.final_hidden) {
    hy.push_back(std::move(std::get<0>(hidden)));
    cy.push_back(std::move(std::get<1>(hidden)));
  }

  return std::make_tuple(result.outputs, at::stack(hy, 0), at::stack(cy, 0));
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////
// PUBLIC FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

#define ONE_HIDDEN_RNN(NAME, CELL)                                             \
DEFINE_DISPATCH(NAME##_cudnn_stub);                                            \
DEFINE_DISPATCH(NAME##_packed_cudnn_stub);                                     \
REGISTER_NO_CPU_DISPATCH(NAME##_cudnn_stub, rnn_fn);                           \
REGISTER_NO_CPU_DISPATCH(NAME##_packed_cudnn_stub, rnn_packed_fn);             \
                                                                               \
std::tuple<Tensor, Tensor> NAME(                                               \
      const Tensor& _input, const Tensor& hx,                                  \
      TensorList _params, bool has_biases,                                     \
      int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) { \
  if (at::cudnn_is_acceptable(_input)) {                                       \
    Tensor output, hy;                                                         \
    NAME##_cudnn_stub(_input.type().device_type(), output, hy, _input, hx, _params, has_biases, \
            num_layers, dropout_p, train, bidirectional, batch_first);         \
    return std::make_tuple(output, hy);                                        \
  }                                                                            \
  check_device(_input, _params, hx);                                        \
  auto input = batch_first ? _input.transpose(0, 1) : _input;                  \
  auto params = gather_params(_params, has_biases);                            \
  auto results = _rnn_impl_with_concat<CELL, FullLayer, FullBidirectionalLayer>( \
          input, params, hx.unbind(0), num_layers, dropout_p, train, bidirectional); \
  if (batch_first) {                                                           \
    std::get<0>(results) = std::get<0>(results).transpose(0, 1);               \
  }                                                                            \
  return results;                                                              \
}                                                                              \
                                                                               \
std::tuple<Tensor, Tensor> NAME(                                               \
      const Tensor& data, const Tensor& batch_sizes, const Tensor& hx,         \
      TensorList _params, bool has_biases,                                     \
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {  \
  if (at::cudnn_is_acceptable(data)) {                                         \
    Tensor output, hy;                                                         \
    NAME##_packed_cudnn_stub(data.type().device_type(), output, hy, data, batch_sizes, hx, \
            _params, has_biases, num_layers, dropout_p, train, bidirectional); \
    return std::make_tuple(output, hy);                                        \
  }                                                                            \
  PackedSequence input { data, batch_sizes };                                  \
  auto params = gather_params(_params, has_biases);                            \
  auto result = _rnn_impl_with_concat<CELL, PackedLayer, PackedBidirectionalLayer>( \
          input, params, hx.unbind(0), num_layers, dropout_p, train, bidirectional); \
  auto & packed_output = std::get<0>(result);                                  \
  return std::make_tuple(packed_output.data, std::get<1>(result));             \
}

ONE_HIDDEN_RNN(gru, GRUCell<CellParams>)
using tanf_cell_type = SimpleCell<tanh_f, CellParams>;
ONE_HIDDEN_RNN(rnn_tanh, tanf_cell_type)
using relu_cell_type = SimpleCell<relu_f, CellParams>;
ONE_HIDDEN_RNN(rnn_relu, relu_cell_type);

DEFINE_DISPATCH(lstm_cudnn_stub);
DEFINE_DISPATCH(lstm_packed_cudnn_stub);
REGISTER_NO_CPU_DISPATCH(lstm_cudnn_stub, lstm_fn);
REGISTER_NO_CPU_DISPATCH(lstm_packed_cudnn_stub, lstm_packed_fn);

std::tuple<Tensor, Tensor, Tensor> lstm(
      const Tensor& _input, TensorList hx,
      TensorList _params, bool has_biases,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  AT_CHECK(hx.size() == 2, "lstm expects two hidden states");
  if (at::cudnn_is_acceptable(_input)) {
    Tensor output, hy, cy;
    lstm_cudnn_stub(_input.type().device_type(), output, hy, cy, _input, hx, _params, has_biases,
            num_layers, dropout_p, train, bidirectional, batch_first);
    return std::make_tuple(output, hy, cy);
  }
  check_device(_input, _params, hx);
  auto input = batch_first ? _input.transpose(0, 1) : _input;
  auto params = gather_params(_params, has_biases);
  auto results = _lstm_impl<FullLayer, FullBidirectionalLayer>(
      input, params, hx[0], hx[1], num_layers, dropout_p, train, bidirectional);
  if (batch_first) {
    std::get<0>(results) = std::get<0>(results).transpose(0, 1);
  }
  return results;
}

std::tuple<Tensor, Tensor, Tensor> lstm(
      const Tensor& data, const Tensor& batch_sizes, TensorList hx,
      TensorList _params, bool has_biases,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {
  AT_CHECK(hx.size() == 2, "lstm expects two hidden states");
  if (at::cudnn_is_acceptable(data)) {
    Tensor output, hy, cy;
    lstm_packed_cudnn_stub(data.type().device_type(), output, hy, cy, data, batch_sizes, hx,
            _params, has_biases, num_layers, dropout_p, train, bidirectional);
    return std::make_tuple(output, hy, cy);
  }
  PackedSequence input { data, batch_sizes };
  auto params = gather_params(_params, has_biases);
  auto result = _lstm_impl<PackedLayer, PackedBidirectionalLayer>(
      input, params, hx[0], hx[1], num_layers, dropout_p, train, bidirectional);
  auto & packed_output = std::get<0>(result);
  return std::make_tuple(packed_output.data, std::get<1>(result), std::get<2>(result));
}

std::tuple<Tensor, Tensor> lstm_cell(
    const Tensor& input, TensorList hx,
    const Tensor& w_ih, const Tensor& w_hh, const Tensor& b_ih, const Tensor& b_hh) {
  AT_CHECK(hx.size() == 2, "lstm_cell expects two hidden states");
  return LSTMCell<CellParams>{}(input, std::make_tuple(hx[0], hx[1]), CellParams{w_ih, w_hh, b_ih, b_hh});
}

Tensor gru_cell(
    const Tensor& input, const Tensor& hx,
    const Tensor& w_ih, const Tensor& w_hh, const Tensor& b_ih, const Tensor& b_hh) {
  return GRUCell<CellParams>{}(input, hx, CellParams{w_ih, w_hh, b_ih, b_hh});
}

Tensor rnn_tanh_cell(
    const Tensor& input, const Tensor& hx,
    const Tensor& w_ih, const Tensor& w_hh, const Tensor& b_ih, const Tensor& b_hh) {
  return SimpleCell<tanh_f, CellParams>{}(input, hx, CellParams{w_ih, w_hh, b_ih, b_hh});
}

Tensor rnn_relu_cell(
    const Tensor& input, const Tensor& hx,
    const Tensor& w_ih, const Tensor& w_hh, const Tensor& b_ih, const Tensor& b_hh) {
  return SimpleCell<relu_f, CellParams>{}(input, hx, CellParams{w_ih, w_hh, b_ih, b_hh});
}

// Quantized implementations
//
// These implementations use FBGEMM to do the i2h and h2h linear layers with
// an int8 quantized weight. This is advantageous in small-batch-size scenarios
// where runtime is dominated by memory fetches of the weight matrix.

std::tuple<Tensor, Tensor, Tensor> quantized_lstm(
      const Tensor& _input, TensorList hx,
      TensorList _params, bool has_biases,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  AT_CHECK(hx.size() == 2, "lstm expects two hidden states");
  if (at::cudnn_is_acceptable(_input)) {
    Tensor output, hy, cy;
    lstm_cudnn_stub(_input.type().device_type(), output, hy, cy, _input, hx, _params, has_biases,
            num_layers, dropout_p, train, bidirectional, batch_first);
    return std::make_tuple(output, hy, cy);
  }
  check_device(_input, _params, hx);
  auto input = batch_first ? _input.transpose(0, 1) : _input;
  AT_CHECK(has_biases, "quantized LSTM requires biases");
  auto params = gather_quantized_params(_params);
  auto results = _lstm_impl<FullLayer, FullBidirectionalLayer>(
      input, params, hx[0], hx[1], num_layers, dropout_p, train, bidirectional);
  if (batch_first) {
    std::get<0>(results) = std::get<0>(results).transpose(0, 1);
  }
  return results;
}

#define DEFINE_QUANTIZED_RNN_CELL(name, hx_type, cell_type, return_type, prepare_hx_fn) \
return_type name( \
    const Tensor& input, \
    hx_type hx, \
    const Tensor& w_ih, \
    const Tensor& w_hh, \
    const Tensor& b_ih, \
    const Tensor& b_hh, \
    const Tensor& packed_ih, \
    const Tensor& packed_hh, \
    const Tensor& col_offsets_ih, \
    const Tensor& col_offsets_hh, \
    const Scalar scale_ih, \
    const Scalar scale_hh, \
    const Scalar zero_point_ih, \
    const Scalar zero_point_hh) { \
  QuantizedCellParams params( \
      w_ih, \
      w_hh, \
      b_ih, \
      b_hh, \
      packed_ih, \
      packed_hh, \
      col_offsets_ih, \
      col_offsets_hh, \
      scale_ih, \
      scale_hh, \
      zero_point_ih, \
      zero_point_hh); \
  return cell_type{}( \
      input, prepare_hx_fn(hx), params); \
}

// Quantized LSTM cell
using quantized_lstm_cell_type = LSTMCell<QuantizedCellParams>;
using quantized_lstm_return_type = std::tuple<Tensor, Tensor>;
std::tuple<Tensor, Tensor> prepare_quantized_lstm_hx(TensorList hx) {
  return std::make_tuple(hx[0], hx[1]);
}
DEFINE_QUANTIZED_RNN_CELL(quantized_lstm_cell, TensorList, quantized_lstm_cell_type, quantized_lstm_return_type, prepare_quantized_lstm_hx);

// Helpers for simpler cells
using simple_hx_type = const Tensor&;
simple_hx_type prepare_quantized_hx(simple_hx_type hx) {
  return hx;
}

// Quantized GRU cell
using quantized_gru_cell_type = GRUCell<QuantizedCellParams>;
DEFINE_QUANTIZED_RNN_CELL(quantized_gru_cell, simple_hx_type, quantized_gru_cell_type, Tensor, prepare_quantized_hx);

// Quantized RNN w/ ReLU cell
using quantized_rnn_relu_cell_type = SimpleCell<relu_f, QuantizedCellParams>;
DEFINE_QUANTIZED_RNN_CELL(quantized_rnn_relu_cell, simple_hx_type, quantized_rnn_relu_cell_type, Tensor, prepare_quantized_hx);

// Quantized RNN w/ tanh cell
using quantized_rnn_tanh_cell_type = SimpleCell<tanh_f, QuantizedCellParams>;
DEFINE_QUANTIZED_RNN_CELL(quantized_rnn_tanh_cell, simple_hx_type, quantized_rnn_tanh_cell_type, Tensor, prepare_quantized_hx);

}}  // namespace at::native
