import { zeros } from "../array";
import { LayerOptions } from "../type/LayerOptions";
import { Vol } from "../vol";
import { BaseLayer } from "./BaseLayer";

interface FullyConnLayerOptions extends LayerOptions {
  out_depth?: number;
  num_neurons?: number;
  l1_decay_mul?: number;
  l2_decay_mul?: number;
  in_sx?: number;
  in_sy?: number;
  in_depth?: number;
  bias_pref?: number;
  out_sx?: number;
  out_sy?: number;
  layer_type?: string;
  num_inputs?: number;
  filters?: number;
  biases?: Record<string, any>;
}

export class FullyConnLayer implements BaseLayer {
  out_depth: number;
  l1_decay_mul: number;
  l2_decay_mul: number;
  num_inputs: number;
  out_sx: number;
  out_sy: number;
  filters: Vol[];
  biases: Vol;
  in_act: Vol;
  out_act: Vol;
  layer_type: string = "fc";

  constructor(opt: FullyConnLayerOptions) {
    opt = opt || {};

    // required
    // ok fine we will allow 'filters' as the word as well
    this.out_depth =
      (typeof opt.num_neurons !== "undefined"
        ? opt.num_neurons
        : opt.filters) || NaN;

    // optional
    this.l1_decay_mul =
      typeof opt.l1_decay_mul !== "undefined" ? opt.l1_decay_mul : 0.0;
    this.l2_decay_mul =
      typeof opt.l2_decay_mul !== "undefined" ? opt.l2_decay_mul : 1.0;

    // computed
    this.num_inputs =
      (opt.in_sx || NaN) * (opt.in_sy || NaN) * (opt.in_depth || NaN);
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = "fc";

    // initializations
    var bias = typeof opt.bias_pref !== "undefined" ? opt.bias_pref : 0.0;
    this.filters = [];
    for (var i = 0; i < this.out_depth; i++) {
      this.filters.push(new Vol(1, 1, this.num_inputs));
    }
    this.biases = new Vol(1, 1, this.out_depth, bias);
  }

  forward(V: Vol, is_training: boolean) {
    this.in_act = V;
    const A = new Vol(1, 1, this.out_depth, 0.0);
    const Vw = V.w;
    for (let i = 0; i < this.out_depth; i++) {
      let a = 0.0;
      const wi = this.filters[i].w;
      for (let d = 0; d < this.num_inputs; d++) {
        a += Vw[d] * wi[d]; // for efficiency use Vols directly for now
      }
      a += this.biases.w[i];
      A.w[i] = a;
    }
    this.out_act = A;
    return this.out_act;
  }

  backward() {
    const V = this.in_act;
    V.dw = zeros(V.w.length); // zero out the gradient in input Vol

    // compute gradient wrt weights and data
    for (let i = 0; i < this.out_depth; i++) {
      const tfi = this.filters[i];
      const chain_grad = this.out_act.dw[i];
      for (let d = 0; d < this.num_inputs; d++) {
        V.dw[d] += tfi.w[d] * chain_grad; // grad wrt input data
        tfi.dw[d] += V.w[d] * chain_grad; // grad wrt params
      }
      this.biases.dw[i] += chain_grad;
    }
  }

  getParamsAndGrads() {
    const response: Array<{
      params: number[] | Float64Array;
      grads: number[] | Float64Array;
      l1_decay_mul: number;
      l2_decay_mul: number;
    }> = [];

    for (let i = 0; i < this.out_depth; i++) {
      response.push({
        params: this.filters[i].w,
        grads: this.filters[i].dw,
        l1_decay_mul: this.l1_decay_mul,
        l2_decay_mul: this.l2_decay_mul,
      });
    }
    response.push({
      params: this.biases.w,
      grads: this.biases.dw,
      l1_decay_mul: 0.0,
      l2_decay_mul: 0.0,
    });
    return response;
  }

  toJSON() {
    const json: Record<string, any> = {};
    json.out_depth = this.out_depth;
    json.out_sx = this.out_sx;
    json.out_sy = this.out_sy;
    json.layer_type = this.layer_type;
    json.num_inputs = this.num_inputs;
    json.l1_decay_mul = this.l1_decay_mul;
    json.l2_decay_mul = this.l2_decay_mul;
    json.filters = [];
    for (let i = 0; i < this.filters.length; i++) {
      json.filters.push(this.filters[i].toJSON());
    }
    json.biases = this.biases.toJSON();
    return json;
  }

  fromJSON(json: Record<string, any>) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type;
    this.num_inputs = json.num_inputs;
    this.l1_decay_mul =
      typeof json.l1_decay_mul !== "undefined" ? json.l1_decay_mul : 1.0;
    this.l2_decay_mul =
      typeof json.l2_decay_mul !== "undefined" ? json.l2_decay_mul : 1.0;
    this.filters = [];
    for (let i = 0; i < json.filters.length; i++) {
      const v = new Vol(0, 0, 0, 0);
      v.fromJSON(json.filters[i]);
      this.filters.push(v);
    }
    this.biases = new Vol(0, 0, 0, 0);
    this.biases.fromJSON(json.biases);
  }
}
