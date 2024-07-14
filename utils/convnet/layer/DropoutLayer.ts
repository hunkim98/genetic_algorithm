import { zeros } from "../array";
import { LayerOptions } from "../type/LayerOptions";
import { Vol } from "../vol";
import { BaseLayer } from "./BaseLayer";

interface DropoutLayerOptions extends LayerOptions {
  in_sx?: number;
  in_sy?: number;
  in_depth?: number;
  drop_prob?: number;
}

export class DropoutLayer implements BaseLayer {
  out_sx: number;
  out_sy: number;
  out_depth: number;
  drop_prob: number;
  dropped: number[] | Float64Array;
  in_act?: Vol;
  out_act?: Vol;
  layer_type: string = "dropout";

  constructor(opt: DropoutLayerOptions) {
    opt = opt || {};

    // computed
    this.out_sx = opt.in_sx || NaN;
    this.out_sy = opt.in_sy || NaN;
    this.out_depth = opt.in_depth || NaN;
    this.layer_type = "dropout";
    this.drop_prob = typeof opt.drop_prob !== "undefined" ? opt.drop_prob : 0.5;
    this.dropped = zeros(this.out_sx * this.out_sy * this.out_depth);
  }

  forward(V: Vol, is_training?: boolean) {
    this.in_act = V;
    if (typeof is_training === "undefined") {
      is_training = false;
    } // default is prediction mode
    var V2 = V.clone();
    var N = V.w.length;
    if (is_training) {
      // do dropout
      for (var i = 0; i < N; i++) {
        if (Math.random() < this.drop_prob) {
          V2.w[i] = 0;
          this.dropped[i] = 1; // true
        } // drop!
        else {
          this.dropped[i] = 0; // false
        }
      }
    } else {
      // scale the activations during prediction
      for (var i = 0; i < N; i++) {
        V2.w[i] *= this.drop_prob;
      }
    }
    this.out_act = V2;
    return this.out_act; // dummy identity function for now
  }

  backward() {
    const V = this.in_act!; // we need to set dw of this
    const chain_grad = this.out_act!;
    const N = V.w.length;
    V.dw = zeros(N); // zero out gradient wrt data
    for (let i = 0; i < N; i++) {
      if (!this.dropped[i]) {
        V.dw[i] = chain_grad.dw[i]; // copy over the gradient
      }
    }
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
    json.drop_prob = this.drop_prob;
    return json;
  }

  fromJSON(json: Record<string, any>) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type;
    this.drop_prob = json.drop_prob;
  }
}
