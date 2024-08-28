import * as p5 from "p5";
import { FACTOR, REF_H, REF_W } from "./constant";

// conversion to pixels
export function toX(x: number): number {
  return (x + REF_W / 2) * FACTOR;
}
export function toP(x: number): number {
  return x * FACTOR;
}
export function toY(p: p5, y: number): number {
  return p.height - y * FACTOR;
}

export function getRandom(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

export function getRandomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min)) + min;
}

export function getRandomColor(p: p5, alpha: number): p5.Color {
  var c = p.color(
    getRandomInt(127, 255),
    getRandomInt(127, 255),
    getRandomInt(127, 255),
    alpha ? alpha : 0
  );
  return c;
}

const cosTable = new Array(360);
const sinTable = new Array(360);
const PI = Math.PI;

// pre compute sine and cosine values to the nearest degree
for (let i = 0; i < 360; i++) {
  cosTable[i] = Math.cos((i / 360) * 2 * PI);
  sinTable[i] = Math.sin((i / 360) * 2 * PI);
}

export function fastSin(xDeg: number) {
  const deg = Math.round(xDeg);
  if (deg >= 0) {
    return sinTable[deg % 360];
  }
  return -sinTable[-deg % 360];
}

export function fastCos(xDeg: number) {
  const deg = Math.round(Math.abs(xDeg));
  return cosTable[deg % 360];
}
