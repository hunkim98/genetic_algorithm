import { LayerOptions } from "../type/LayerOptions";
import { Vol } from "../vol";
import { BaseLayer } from "./BaseLayer";

interface InputLayerOptions extends LayerOptions {
  out_sx?: number;
  out_sy?: number;
  out_depth?: number;
  in_sx?: number;
  in_sy?: number;
  in_depth?: number;
  layer_type?: string;
}

export class InputLayer implements BaseLayer {
  out_sx: number;
  out_sy: number;
  out_depth: number;
  in_act: Vol;
  out_act: Vol;
  layer_type: string = "input";

  constructor(opt: InputLayerOptions) {
    opt = opt || {};

    // this is a bit silly but lets allow people to specify either ins or outs
    this.out_sx =
      typeof opt.out_sx !== "undefined" ? opt.out_sx : opt.in_sx || NaN;
    this.out_sy =
      typeof opt.out_sy !== "undefined" ? opt.out_sy : opt.in_sy || NaN;
    this.out_depth =
      typeof opt.out_depth !== "undefined"
        ? opt.out_depth
        : opt.in_depth || NaN;
    this.layer_type = "input";
  }

  forward(V: Vol, isTraining?: boolean) {
    this.in_act = V;
    this.out_act = V;
    return this.out_act; // dummy identity function for now
  }

  backward() {}

  getParamsAndGrads() {
    return [];
  }

  toJSON() {
    const json: InputLayerOptions = {
      out_depth: this.out_depth,
      out_sx: this.out_sx,
      out_sy: this.out_sy,
      layer_type: this.layer_type,
    };
    return json;
  }

  fromJSON(json: InputLayerOptions) {
    this.out_depth = json.out_depth!;
    this.out_sx = json.out_sx!;
    this.out_sy = json.out_sy!;
    this.layer_type = json.layer_type!;
  }
}
