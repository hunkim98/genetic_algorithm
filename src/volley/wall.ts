import p5 from "p5";
import { toP, toX, toY } from "./utils";

export class Wall {
  private x: number;
  private y: number;
  private w: number;
  private h: number;
  private c: p5.Color;
  private p: p5;

  constructor(p: p5, x: number, y: number, w: number, h: number) {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.c = p.color(0, 200, 50, 128);
    this.p = p;
  }

  display(): void {
    this.p.noStroke();
    this.p.fill(255);
    this.p.rect(
      toX(this.x - this.w / 2),
      toY(this.p, this.y + this.h / 2),
      toP(this.w),
      toP(this.h)
    );
    this.p.fill(this.c);
    this.p.rect(
      toX(this.x - this.w / 2),
      toY(this.p, this.y + this.h / 2),
      toP(this.w),
      toP(this.h)
    );
  }
}
