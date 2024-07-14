import { zeros } from "../array";
import { LayerOptions } from "../type/LayerOptions";
import { Vol } from "../vol";
import { BaseLayer } from "./BaseLayer";

interface SoftmaxLayerOptions extends LayerOptions {
  in_sx?: number;
  in_sy?: number;
  in_depth?: number;
}

export class SoftmaxLayer implements BaseLayer {
  num_inputs: number;
  out_depth: number;
  out_sx: number;
  out_sy: number;
  in_act?: Vol;
  out_act?: Vol;
  es?: number[] | Float64Array;
  layer_type: string = "softmax";

  constructor(opt: SoftmaxLayerOptions) {
    opt = opt || {};

    // computed
    this.num_inputs =
      (opt.in_sx || NaN) * (opt.in_sy || NaN) * (opt.in_depth || NaN);
    this.out_depth = this.num_inputs;
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = "softmax";
  }

  forward(V: Vol, is_training?: boolean) {
    this.in_act = V;

    const A = new Vol(1, 1, this.out_depth, 0.0);

    // compute max activation
    const as = V.w;
    let amax = V.w[0];
    for (let i = 1; i < this.out_depth; i++) {
      if (as[i] > amax) amax = as[i];
    }

    // compute exponentials (carefully to not blow up)
    const es = zeros(this.out_depth);
    let esum = 0.0;
    for (let i = 0; i < this.out_depth; i++) {
      const e = Math.exp(as[i] - amax);
      esum += e;
      es[i] = e;
    }

    // normalize and output to sum to one
    for (let i = 0; i < this.out_depth; i++) {
      es[i] /= esum;
      A.w[i] = es[i];
    }

    this.es = es; // save these for backprop
    this.out_act = A;
    return this.out_act;
  }

  backward(y: number) {
    // compute and accumulate gradient wrt weights and bias of this layer
    const x = this.in_act!;
    x.dw = zeros(x.w.length); // zero out the gradient of input Vol

    for (var i = 0; i < this.out_depth; i++) {
      const indicator = i === y ? 1.0 : 0.0;
      const mul = -(indicator - this.es![i]);
      x.dw[i] = mul;
    }

    // loss is the class negative log likelihood
    return -Math.log(this.es![y]);
  }

  getParamsAndGrads() {
    return [];
  }

  toJSON() {
    const json: Record<string, any> = {};
    json.out_depth = this.out_depth;
    json.out_sx = this.out_sx;
    json.out_sy = this.out_sy;
    json.layer_type = this.layer_type;
    json.num_inputs = this.num_inputs;
    return json;
  }

  fromJSON(json: Record<string, any>) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type;
    this.num_inputs = json.num_inputs;
  }
}
