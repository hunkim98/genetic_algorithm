import { Vol } from "../vol";

export abstract class BaseLayer {
  abstract out_sx: number;
  abstract out_sy: number;
  abstract out_depth: number;
  abstract in_act: Vol;
  abstract out_act: Vol;
  abstract layer_type: string;

  abstract forward(V: Vol, is_training?: boolean): Vol;
  abstract backward(): void;
  abstract getParamsAndGrads(): Array<{
    params: number[] | Float64Array;
    grads: number[] | Float64Array;
    l1_decay_mul: number;
    l2_decay_mul: number;
  }>;
  abstract toJSON(): Record<string, any> | JSON;
  abstract fromJSON(json: Record<string, any>): void;
}
