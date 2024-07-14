import { BaseLayer } from "./layer/BaseLayer";
import { ConvLayer } from "./layer/ConvLayer";
import { DropoutLayer } from "./layer/DropoutLayer";
import { FullyConnLayer } from "./layer/FullyConnLayer";
import { InputLayer } from "./layer/InputLayer";
import { LocalResponseNormalizationLayer } from "./layer/LocalResponseNormalizationLayer";
import { MaxoutLayer } from "./layer/MaxoutLayer";
import { PoolLayer } from "./layer/PoolLayer";
import { QuadTransformLayer } from "./layer/QuadTransformLayer";
import { RegressionLayer } from "./layer/RegressionLayer";
import { ReluLayer } from "./layer/ReluLayer";
import { SigmoidLayer } from "./layer/SigmoidLayer";
import { SoftmaxLayer } from "./layer/SoftmaxLayer";
import { SVMLayer } from "./layer/SVMLayer";
import { TanhLayer } from "./layer/TanhLayer";

class Net {
  layers: BaseLayer[];
  constructor(options) {
    this.layers = [];
  }

  makeLayers(
    defs: Array<{
      type: string;
      [key: string]: any;
    }>
  ) {
    if (defs.length < 2) {
      console.log("ERROR! For now at least have input and softmax layers.");
    }
    if (defs[0].type !== "input") {
      console.log("ERROR! For now first layer should be input.");
    }

    const desugar = (
      defs: Array<{
        type: string;
        [key: string]: any;
      }>
    ) => {
      const newDefs: {
        type: string;
        num_neurons?: number;
        num_classes?: number;
        bias_pref?: number;
        activation?: string;
        tensor?: boolean;
        group_size?: number;
        drop_prob?: number;
      }[] = [];

      for (let i = 0; i < defs.length; i++) {
        const def = defs[i];

        if (def.type === "softmax" || def.type === "svm") {
          newDefs.push({ type: "fc", num_neurons: def.num_classes });
        }

        if (def.type === "regression") {
          newDefs.push({ type: "fc", num_neurons: def.num_neurons });
        }

        if (
          (def.type === "fc" || def.type === "conv") &&
          typeof def.bias_pref === "undefined"
        ) {
          def.bias_pref = 0.0;
          if (
            typeof def.activation !== "undefined" &&
            def.activation === "relu"
          ) {
            def.bias_pref = 0.1;
          }
        }

        if (typeof def.tensor !== "undefined") {
          if (def.tensor) {
            newDefs.push({ type: "quadtransform" });
          }
        }

        newDefs.push(def);

        if (typeof def.activation !== "undefined") {
          if (def.activation === "relu") {
            newDefs.push({ type: "relu" });
          } else if (def.activation === "sigmoid") {
            newDefs.push({ type: "sigmoid" });
          } else if (def.activation === "tanh") {
            newDefs.push({ type: "tanh" });
          } else if (def.activation === "maxout") {
            const gs = def.group_size !== "undefined" ? def.group_size : 2;
            newDefs.push({ type: "maxout", group_size: gs });
          } else {
            console.log("ERROR unsupported activation " + def.activation);
          }
        }
        if (typeof def.drop_prob !== "undefined" && def.type !== "dropout") {
          newDefs.push({ type: "dropout", drop_prob: def.drop_prob });
        }
      }
      return newDefs;
    };
    defs = desugar(defs);

    this.layers = [];
    for (let i = 0; i < defs.length; i++) {
      const def = defs[i];
      if (i > 0) {
        const prev = this.layers[i - 1];
        def.in_sx = prev.out_sx;
        def.in_sy = prev.out_sy;
        def.in_depth = prev.out_depth;
      }

      switch (def.type) {
        case "fc":
          this.layers.push(new FullyConnLayer(def));
          break;
        case "lrn":
          this.layers.push(new LocalResponseNormalizationLayer(def));
          break;
        case "dropout":
          this.layers.push(new DropoutLayer(def));
          break;
        case "input":
          this.layers.push(new InputLayer(def));
          break;
        case "relu":
          this.layers.push(new ReluLayer(def));
          break;
        case "sigmoid":
          this.layers.push(new SigmoidLayer(def));
          break;
        case "tanh":
          this.layers.push(new TanhLayer(def));
          break;
        case "maxout":
          this.layers.push(new MaxoutLayer(def));
          break;
        case "softmax":
          this.layers.push(new SoftmaxLayer(def));
          break;
        case "regression":
          this.layers.push(new RegressionLayer(def));
          break;
        case "conv":
          this.layers.push(new ConvLayer(def));
          break;
        case "pool":
          this.layers.push(new PoolLayer(def));
          break;
        case "quadtransform":
          this.layers.push(new QuadTransformLayer(def));
          break;
        case "svm":
          this.layers.push(new SVMLayer(def));
          break;
        default:
          console.log("ERROR: UNRECOGNIZED LAYER TYPE!");
      }
    }
  }

  forward(V, is_training = false) {
    let act = this.layers[0].forward(V, is_training);
    for (let i = 1; i < this.layers.length; i++) {
      act = this.layers[i].forward(act, is_training);
    }
    return act;
  }

  getCostLoss(V, y) {
    this.forward(V, false);
    const N = this.layers.length;
    // const loss = this.layers[N - 1].backward();
    const loss = this.layers[N - 1].backward(y);
    return loss;
  }

  backward(y) {
    const N = this.layers.length;
    // const loss = this.layers[N - 1].backward();
    const loss = this.layers[N - 1].backward(y);
    for (let i = N - 2; i >= 0; i--) {
      this.layers[i].backward();
    }
    return loss;
  }

  getParamsAndGrads() {
    const response: Array<{
      params: number[] | Float64Array;
      grads: number[] | Float64Array;
      l1_decay_mul: number;
      l2_decay_mul: number;
    }> = [];
    for (let i = 0; i < this.layers.length; i++) {
      const layerResponse = this.layers[i].getParamsAndGrads();
      for (let j = 0; j < layerResponse.length; j++) {
        response.push(layerResponse[j]);
      }
    }
    return response;
  }

  getPrediction() {
    const S = this.layers[this.layers.length - 1];
    const p = S.out_act.w;
    let maxv = p[0];
    let maxi = 0;
    for (let i = 1; i < p.length; i++) {
      if (p[i] > maxv) {
        maxv = p[i];
        maxi = i;
      }
    }
    return maxi;
  }

  toJSON() {
    const json = {} as {
      layers: any[];
    };
    json.layers = [];
    for (let i = 0; i < this.layers.length; i++) {
      json.layers.push(this.layers[i].toJSON());
    }
    return json;
  }

  fromJSON(json) {
    this.layers = [];
    for (let i = 0; i < json.layers.length; i++) {
      const Lj = json.layers[i];
      const t = Lj.layer_type;
      let L;
      if (t === "input") {
        L = new InputLayer({});
      }
      if (t === "relu") {
        L = new ReluLayer({});
      }
      if (t === "sigmoid") {
        L = new SigmoidLayer({});
      }
      if (t === "tanh") {
        L = new TanhLayer({});
      }
      if (t === "dropout") {
        L = new DropoutLayer({});
      }
      if (t === "conv") {
        L = new ConvLayer({});
      }
      if (t === "pool") {
        L = new PoolLayer({});
      }
      if (t === "lrn") {
        L = new LocalResponseNormalizationLayer({});
      }
      if (t === "softmax") {
        L = new SoftmaxLayer({});
      }
      if (t === "regression") {
        L = new RegressionLayer({});
      }
      if (t === "fc") {
        L = new FullyConnLayer({});
      }
      if (t === "maxout") {
        L = new MaxoutLayer({});
      }
      if (t === "quadtransform") {
        L = new QuadTransformLayer({});
      }
      if (t === "svm") {
        L = new SVMLayer({});
      }
      L.fromJSON(Lj);
      this.layers.push(L);
    }
  }
}
