// Vol is the basic building block of all data in a net.
// it is essentially just a 3D volume of numbers, with a
// width (sx), height (sy), and depth (depth).
// it is used to hold data for all filters, all volumes,
// all weights, and also stores all gradients w.r.t.
// the data. c is optionally a value to initialize the volume

import { zeros } from "./array";
import { randn } from "./random";

// with. If c is missing, fills the Vol with random numbers.
export class Vol {
  sx: number;
  sy: number;
  depth: number;
  w: number[] | Float64Array;
  dw: number[] | Float64Array;

  // sx is width, sy is height, and depth is the number of channels
  constructor(sx: number, sy: number, depth: number, c?: number) {
    // this is how you check if a variable is an array. Oh, Javascript :)
    if (Array.isArray(sx)) {
      // we were given a list in sx, assume 1D volume and fill it up
      this.sx = 1;
      this.sy = 1;
      this.depth = sx.length;
      // we have to do the following copy because we want to use
      // fast typed arrays, not an ordinary javascript array
      this.w = zeros(this.depth);
      this.dw = zeros(this.depth);
      for (let i = 0; i < this.depth; i++) {
        this.w[i] = sx[i];
      }
    } else {
      // we were given dimensions of the vol
      this.sx = sx;
      this.sy = sy;
      this.depth = depth;
      const n = sx * sy * depth;
      this.w = zeros(n);
      this.dw = zeros(n);
      if (typeof c === "undefined") {
        // weight normalization is done to equalize the output
        // variance of every neuron, otherwise neurons with a lot
        // of incoming connections have outputs of larger variance
        const scale = Math.sqrt(1.0 / (sx * sy * depth));
        for (let i = 0; i < n; i++) {
          this.w[i] = randn(0.0, scale);
        }
      } else {
        for (let i = 0; i < n; i++) {
          this.w[i] = c;
        }
      }
    }
  }

  /**
   *
   * @param x x coordinate
   * @param y y coordinate
   * @param d depth
   * @returns the value at the specified coordinates
   */
  get(x: number, y: number, d: number): number {
    const ix = (this.sx * y + x) * this.depth + d;
    return this.w[ix];
  }

  /**
   *
   * @param x x coordinate
   * @param y y coordinate
   * @param d depth
   * @param v the value to set
   */
  set(x: number, y: number, d: number, v: number): void {
    const ix = (this.sx * y + x) * this.depth + d;
    this.w[ix] = v;
  }

  /**
   *
   * @param x x coordinate
   * @param y y coordinate
   * @param d depth
   * @param v the value to add
   */
  add(x: number, y: number, d: number, v: number): void {
    const ix = (this.sx * y + x) * this.depth + d;
    this.w[ix] += v;
  }

  /**
   *
   * @param x x coordinate
   * @param y y coordinate
   * @param d depth
   * @returns the gradient at the specified coordinates
   */
  get_grad(x: number, y: number, d: number): number {
    const ix = (this.sx * y + x) * this.depth + d;
    return this.dw[ix];
  }

  /**
   *
   * @param x x coordinate
   * @param y y coordinate
   * @param d depth
   * @param v the value to set the gradient to
   */
  set_grad(x: number, y: number, d: number, v: number): void {
    const ix = (this.sx * y + x) * this.depth + d;
    this.dw[ix] = v;
  }

  /**
   *
   * @param x x coordinate
   * @param y y coordinate
   * @param d depth
   * @param v the value to add to the gradient
   */
  add_grad(x: number, y: number, d: number, v: number): void {
    const ix = (this.sx * y + x) * this.depth + d;
    this.dw[ix] += v;
  }

  /**
   *
   * @returns a new volume that is a clone of the current volume
   */
  cloneAndZero(): Vol {
    return new Vol(this.sx, this.sy, this.depth, 0.0);
  }

  /**
   * @description returns a new volume that is a clone of the current volume (with the same weights as the current volume)
   * @returns a new volume that is a clone of the current volume
   */
  clone(): Vol {
    const V = new Vol(this.sx, this.sy, this.depth, 0.0);
    const n = this.w.length;
    for (let i = 0; i < n; i++) {
      V.w[i] = this.w[i];
    }
    return V;
  }

  /**
   * @description add the current volume to the input volume
   * @param V the volume to add to the current volume
   */
  addFrom(V: Vol): void {
    for (let k = 0; k < this.w.length; k++) {
      this.w[k] += V.w[k];
    }
  }

  /**
   * @description add the current volume to the input volume scaled by a
   * @param V the volume to add to the current volume
   * @param a the scale factor
   */
  addFromScaled(V: Vol, a: number): void {
    for (let k = 0; k < this.w.length; k++) {
      this.w[k] += a * V.w[k];
    }
  }

  /**
   * @description set the volume to the input volume
   * @param V the volume to set the current volume to
   */
  setConst(a: number): void {
    for (let k = 0; k < this.w.length; k++) {
      this.w[k] = a;
    }
  }

  /**
   * @description set the volume to the input volume
   * @param V the volume to set the current volume to
   */
  toJSON(): any {
    // todo: we may want to only save d most significant digits to save space
    const json: any = {};
    json.sx = this.sx;
    json.sy = this.sy;
    json.depth = this.depth;
    json.w = this.w;
    return json;
    // we wont back up gradients to save space
  }

  /**
   * @description create a volume from a JSON object
   */
  fromJSON(json: any): void {
    this.sx = json.sx;
    this.sy = json.sy;
    this.depth = json.depth;

    const n = this.sx * this.sy * this.depth;
    this.w = new Array(n);
    this.dw = new Array(n);
    // copy over the elements.
    for (let i = 0; i < n; i++) {
      this.w[i] = json.w[i];
    }
  }
}
