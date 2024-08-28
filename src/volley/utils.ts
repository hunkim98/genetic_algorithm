import * as p5 from "p5";
import { FACTOR, REF_H, REF_W } from "./constant";

// conversion to pixels
export function toX(x: number): number {
  return (x + REF_W / 2) * FACTOR;
}
export function toP(x: number): number {
  return x * FACTOR;
}
export function toY(y: number): number {
  return REF_H - y * FACTOR;
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
