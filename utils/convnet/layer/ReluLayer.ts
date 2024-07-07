import { zeros } from "../array";
import { Vol } from "../vol";
import { BaseLayer } from "./BaseLayer";

interface ReluLayerOptions {
  in_sx?: number;
  in_sy?: number;
  in_depth?: number;
}

export class ReluLayer implements BaseLayer {
  out_sx: number;
  out_sy: number;
  out_depth: number;
  in_act: Vol;
  out_act: Vol;
  layer_type: string = "relu";

  constructor(opt: ReluLayerOptions) {
    opt = opt || {};

    // computed
    this.out_sx = opt.in_sx!;
    this.out_sy = opt.in_sy!;
    this.out_depth = opt.in_depth!;
    this.layer_type = "relu";
  }

  forward(V: Vol, is_training: boolean) {
    this.in_act = V;
    const V2 = V.clone();
    const N = V.w.length;
    const V2w = V2.w;
    for (let i = 0; i < N; i++) {
      if (V2w[i] < 0) V2w[i] = 0; // threshold at 0
    }
    this.out_act = V2;
    return this.out_act;
  }

  backward() {
    const V = this.in_act; // we need to set dw of this
    const V2 = this.out_act;
    const N = V.w.length;
    V.dw = zeros(N); // zero out gradient wrt data
    for (let i = 0; i < N; i++) {
      if (V2.w[i] <= 0) V.dw[i] = 0; // threshold
      else V.dw[i] = V2.dw[i];
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
    return json;
  }

  fromJSON(json: Record<string, any>) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type;
  }
}