import { zeros } from "../array";
import { LayerOptions } from "../type/LayerOptions";
import { Vol } from "../vol";
import { BaseLayer } from "./BaseLayer";

interface SVMLayerOptions extends LayerOptions {
  in_sx?: number;
  in_sy?: number;
  in_depth?: number;
}

export class SVMLayer implements BaseLayer {
  num_inputs: number;
  out_depth: number;
  out_sx: number;
  out_sy: number;
  in_act: Vol;
  out_act: Vol;
  layer_type: string = "svm";

  constructor(opt: SVMLayerOptions) {
    opt = opt || {};

    // computed
    this.num_inputs =
      (opt.in_sx || NaN) * (opt.in_sy || NaN) * (opt.in_depth || NaN);
    this.out_depth = this.num_inputs;
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = "svm";
  }

  forward(V: Vol, is_training?: boolean) {
    this.in_act = V;
    this.out_act = V; // nothing to do, output raw scores
    return V;
  }

  backward(y: number) {
    // compute and accumulate gradient wrt weights and bias of this layer
    const x = this.in_act;
    x.dw = zeros(x.w.length); // zero out the gradient of input Vol

    const yscore = x.w[y]; // score of ground truth
    const margin = 1.0;
    let loss = 0.0;
    for (let i = 0; i < this.out_depth; i++) {
      if (-yscore + x.w[i] + margin > 0) {
        // violating example, apply loss
        // I love hinge loss, by the way. Truly.
        // Seriously, compare this SVM code with Softmax forward AND backprop code above
        // it's clear which one is superior, not only in code, simplicity
        // and beauty, but also in practice.
        x.dw[i] += 1;
        x.dw[y] -= 1;
        loss += -yscore + x.w[i] + margin;
      }
    }

    return loss;
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
