import { zeros } from "../array";
import { LayerOptions } from "../type/LayerOptions";
import { Vol } from "../vol";
import { BaseLayer } from "./BaseLayer";

interface LocalResponseNormalizationLayerOptions extends LayerOptions {
  k?: number;
  n?: number;
  alpha?: number;
  beta?: number;
  in_sx?: number;
  in_sy?: number;
  in_depth?: number;
}

export class LocalResponseNormalizationLayer implements BaseLayer {
  k: number;
  n: number;
  alpha: number;
  beta: number;
  out_sx: number;
  out_sy: number;
  out_depth: number;
  in_act: any;
  out_act: any;
  S_cache_: any;
  layer_type: string = "lrn";

  constructor(opt: LocalResponseNormalizationLayerOptions) {
    opt = opt || {};

    // required
    this.k = opt.k || NaN;
    this.n = opt.n || NaN;
    this.alpha = opt.alpha || NaN;
    this.beta = opt.beta || NaN;

    // computed
    this.out_sx = opt.in_sx || NaN;
    this.out_sy = opt.in_sy || NaN;
    this.out_depth = opt.in_depth || NaN;
    this.layer_type = "lrn";

    // checks
    if (this.n % 2 === 0) {
      console.log("WARNING n should be odd for LRN layer");
    }
  }

  forward(V: Vol, is_training?: boolean) {
    this.in_act = V;

    const A = V.cloneAndZero();
    this.S_cache_ = V.cloneAndZero();
    const n2 = Math.floor(this.n / 2);
    for (let x = 0; x < V.sx; x++) {
      for (let y = 0; y < V.sy; y++) {
        for (let i = 0; i < V.depth; i++) {
          const ai = V.get(x, y, i);

          // normalize in a window of size n
          let den = 0.0;
          for (
            let j = Math.max(0, i - n2);
            j <= Math.min(i + n2, V.depth - 1);
            j++
          ) {
            const aa = V.get(x, y, j);
            den += aa * aa;
          }
          den *= this.alpha / this.n;
          den += this.k;
          this.S_cache_.set(x, y, i, den); // will be useful for backprop
          den = Math.pow(den, this.beta);
          A.set(x, y, i, ai / den);
        }
      }
    }

    this.out_act = A;
    return this.out_act; // dummy identity function for now
  }

  backward() {
    // evaluate gradient wrt data
    const V = this.in_act; // we need to set dw of this
    V.dw = zeros(V.w.length); // zero out gradient wrt data
    const A = this.out_act; // computed in forward pass

    var n2 = Math.floor(this.n / 2);
    for (let x = 0; x < V.sx; x++) {
      for (let y = 0; y < V.sy; y++) {
        for (let i = 0; i < V.depth; i++) {
          const chain_grad = this.out_act.get_grad(x, y, i);
          const S = this.S_cache_.get(x, y, i);
          const SB = Math.pow(S, this.beta);
          const SB2 = SB * SB;

          // normalize in a window of size n
          for (
            let j = Math.max(0, i - n2);
            j <= Math.min(i + n2, V.depth - 1);
            j++
          ) {
            const aj = V.get(x, y, j);
            let g =
              ((-aj * this.beta * Math.pow(S, this.beta - 1) * this.alpha) /
                this.n) *
              2 *
              aj;
            if (j === i) g += SB;
            g /= SB2;
            g *= chain_grad;
            V.add_grad(x, y, j, g);
          }
        }
      }
    }
  }

  getParamsAndGrads() {
    return [];
  }

  toJSON() {
    const json: Record<string, any> = {};
    json.k = this.k;
    json.n = this.n;
    json.alpha = this.alpha; // normalize by size
    json.beta = this.beta;
    json.out_sx = this.out_sx;
    json.out_sy = this.out_sy;
    json.out_depth = this.out_depth;
    json.layer_type = this.layer_type;
    return json;
  }

  fromJSON(json: Record<string, any>) {
    this.k = json.k;
    this.n = json.n;
    this.alpha = json.alpha; // normalize by size
    this.beta = json.beta;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.out_depth = json.out_depth;
    this.layer_type = json.layer_type;
  }
}
