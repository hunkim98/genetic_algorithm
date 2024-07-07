import { zeros } from "../array";
import { Vol } from "../vol";
import { BaseLayer } from "./BaseLayer";

function tanh(x: number): number {
  const y = Math.exp(2 * x);
  return (y - 1) / (y + 1);
}

interface TanhLayerOptions {
  in_sx?: number;
  in_sy?: number;
  in_depth?: number;
}

export class TanhLayer implements BaseLayer {
  out_sx: number;
  out_sy: number;
  out_depth: number;
  in_act: Vol;
  out_act: Vol;
  layer_type: string = "tanh";

  constructor(opt: TanhLayerOptions) {
    opt = opt || {};

    // computed
    this.out_sx = opt.in_sx!;
    this.out_sy = opt.in_sy!;
    this.out_depth = opt.in_depth!;
    this.layer_type = "tanh";
  }

  forward(V: Vol, is_training: boolean) {
    this.in_act = V;
    var V2 = V.cloneAndZero();
    var N = V.w.length;
    for (var i = 0; i < N; i++) {
      V2.w[i] = tanh(V.w[i]);
    }
    this.out_act = V2;
    return this.out_act;
  }

  backward() {
    var V = this.in_act; // we need to set dw of this
    var V2 = this.out_act;
    var N = V.w.length;
    V.dw = zeros(N); // zero out gradient wrt data
    for (var i = 0; i < N; i++) {
      var v2wi = V2.w[i];
      V.dw[i] = (1.0 - v2wi * v2wi) * V2.dw[i];
    }
  }

  getParamsAndGrads() {
    return [];
  }

  toJSON() {
    var json: Record<string, any> = {};
    json.out_depth = this.out_depth;
    json.out_sx = this.out_sx;
    json.out_sy = this.out_sy;
    json.layer_type = this.layer_type;
    return json;
  }

  fromJSON(json: Record<string, any>) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type;
  }
}
