import { zeros } from "../array";
import { LayerOptions } from "../type/LayerOptions";
import { Vol } from "../vol";
import { BaseLayer } from "./BaseLayer";

interface ConvLayerOptions extends LayerOptions {
  filters?: number;
  sx?: number;
  in_depth?: number;
  in_sx?: number;
  in_sy?: number;
  sy?: number;
  stride?: number;
  pad?: number;
  l1_decay_mul?: number;
  l2_decay_mul?: number;
  bias_pref?: number;
}

export class ConvLayer implements BaseLayer {
  out_depth: number;
  sx: number;
  in_depth: number;
  in_sx: number;
  in_sy: number;
  sy: number;
  stride: number;
  pad: number;
  l1_decay_mul: number;
  l2_decay_mul: number;
  out_sx: number;
  out_sy: number;
  filters: Vol[];
  biases: Vol;
  in_act?: Vol;
  out_act?: Vol;
  layer_type: string = "conv";

  constructor(opt: ConvLayerOptions) {
    opt = opt || {};

    // required
    this.out_depth = opt.filters || NaN;
    this.sx = opt.sx || NaN; // filter size. Should be odd if possible, it's cleaner.
    this.in_depth = opt.in_depth || NaN;
    this.in_sx = opt.in_sx || NaN;
    this.in_sy = opt.in_sy || NaN;

    // optional
    this.sy = typeof opt.sy !== "undefined" ? opt.sy : this.sx;
    this.stride = typeof opt.stride !== "undefined" ? opt.stride : 1; // stride at which we apply filters to input volume
    this.pad = typeof opt.pad !== "undefined" ? opt.pad : 0; // amount of 0 padding to add around borders of input volume
    this.l1_decay_mul =
      typeof opt.l1_decay_mul !== "undefined" ? opt.l1_decay_mul : 0.0;
    this.l2_decay_mul =
      typeof opt.l2_decay_mul !== "undefined" ? opt.l2_decay_mul : 1.0;

    // computed
    // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
    // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
    // final application.
    this.out_sx = Math.floor(
      (this.in_sx + this.pad * 2 - this.sx) / this.stride + 1
    );
    this.out_sy = Math.floor(
      (this.in_sy + this.pad * 2 - this.sy) / this.stride + 1
    );
    this.layer_type = "conv";

    // initializations
    var bias = typeof opt.bias_pref !== "undefined" ? opt.bias_pref : 0.0;
    this.filters = [];
    for (let i = 0; i < this.out_depth; i++) {
      this.filters.push(new Vol(this.sx, this.sy, this.in_depth));
    }
    this.biases = new Vol(1, 1, this.out_depth, bias);
  }

  forward(V: Vol, is_training?: boolean) {
    this.in_act = V;

    const A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
    for (let d = 0; d < this.out_depth; d++) {
      const f = this.filters[d];
      let x = -this.pad;
      let y = -this.pad;
      for (let ax = 0; ax < this.out_sx; x += this.stride, ax++) {
        y = -this.pad;
        for (let ay = 0; ay < this.out_sy; y += this.stride, ay++) {
          // convolve centered at this particular location
          // could be bit more efficient, going for correctness first
          let a = 0.0;
          for (let fx = 0; fx < f.sx; fx++) {
            for (let fy = 0; fy < f.sy; fy++) {
              for (let fd = 0; fd < f.depth; fd++) {
                let oy = y + fy; // coordinates in the original input array coordinates
                let ox = x + fx;
                if (oy >= 0 && oy < V.sy && ox >= 0 && ox < V.sx) {
                  //a += f.get(fx, fy, fd) * V.get(ox, oy, fd);
                  // avoid function call overhead for efficiency, compromise modularity :(
                  a +=
                    f.w[(f.sx * fy + fx) * f.depth + fd] *
                    V.w[(V.sx * oy + ox) * V.depth + fd];
                }
              }
            }
          }
          a += this.biases.w[d];
          A.set(ax, ay, d, a);
        }
      }
    }
    this.out_act = A;
    return this.out_act;
  }

  backward() {
    // compute gradient wrt weights, biases and input data
    const V = this.in_act!;
    V.dw = zeros(V.w.length); // zero out gradient wrt bottom data, we're about to fill it
    for (let d = 0; d < this.out_depth; d++) {
      const f = this.filters[d];
      let x = -this.pad;
      let y = -this.pad;
      for (let ax = 0; ax < this.out_sx; x += this.stride, ax++) {
        y = -this.pad;
        for (let ay = 0; ay < this.out_sy; y += this.stride, ay++) {
          // convolve and add up the gradients.
          // could be more efficient, going for correctness first
          const chain_grad = this.out_act!.get_grad(ax, ay, d); // gradient from above, from chain rule
          for (let fx = 0; fx < f.sx; fx++) {
            for (let fy = 0; fy < f.sy; fy++) {
              for (let fd = 0; fd < f.depth; fd++) {
                const oy = y + fy;
                const ox = x + fx;
                if (oy >= 0 && oy < V.sy && ox >= 0 && ox < V.sx) {
                  // forward prop calculated: a += f.get(fx, fy, fd) * V.get(ox, oy, fd);
                  //f.add_grad(fx, fy, fd, V.get(ox, oy, fd) * chain_grad);
                  //V.add_grad(ox, oy, fd, f.get(fx, fy, fd) * chain_grad);

                  // avoid function call overhead and use Vols directly for efficiency
                  const ix1 = (V.sx * oy + ox) * V.depth + fd;
                  const ix2 = (f.sx * fy + fx) * f.depth + fd;
                  f.dw[ix2] += V.w[ix1] * chain_grad;
                  V.dw[ix1] += f.w[ix2] * chain_grad;
                }
              }
            }
          }
          this.biases.dw[d] += chain_grad;
        }
      }
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
        l2_decay_mul: this.l2_decay_mul,
        l1_decay_mul: this.l1_decay_mul,
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
    json.sx = this.sx; // filter size in x, y dims
    json.sy = this.sy;
    json.stride = this.stride;
    json.in_depth = this.in_depth;
    json.out_depth = this.out_depth;
    json.out_sx = this.out_sx;
    json.out_sy = this.out_sy;
    json.layer_type = this.layer_type;
    json.l1_decay_mul = this.l1_decay_mul;
    json.l2_decay_mul = this.l2_decay_mul;
    json.pad = this.pad;
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
    this.sx = json.sx; // filter size in x, y dims
    this.sy = json.sy;
    this.stride = json.stride;
    this.in_depth = json.in_depth; // depth of input volume
    this.filters = [];
    this.l1_decay_mul =
      typeof json.l1_decay_mul !== "undefined" ? json.l1_decay_mul : 1.0;
    this.l2_decay_mul =
      typeof json.l2_decay_mul !== "undefined" ? json.l2_decay_mul : 1.0;
    this.pad = typeof json.pad !== "undefined" ? json.pad : 0;
    for (let i = 0; i < json.filters.length; i++) {
      const v = new Vol(0, 0, 0, 0);
      v.fromJSON(json.filters[i]);
      this.filters.push(v);
    }
    this.biases = new Vol(0, 0, 0, 0);
    this.biases.fromJSON(json.biases);
  }
}
