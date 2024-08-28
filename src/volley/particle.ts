import * as p5 from "p5";
import {
  FRICTION,
  NUDGE,
  REF_H,
  REF_U,
  REF_W,
  REF_WALLHEIGHT,
  REF_WALLWIDTH,
  TIME_STEP,
  WIND_DRAG,
} from "./constant";
import { getRandomColor, toP, toX, toY } from "./utils";

export class Particle {
  loc: p5.Vector;
  prevLoc: p5.Vector;
  v: p5.Vector;
  r: number;
  c: p5.Color;
  p: p5;

  constructor(p: p5, loc?: p5.Vector, v?: p5.Vector, r?: number, c?: p5.Color) {
    this.p = p;
    this.loc =
      loc ||
      new p5.Vector(
        p.random((-REF_W * 1) / 4, (REF_W * 1) / 4),
        p.random(REF_W / 4, (REF_W * 3) / 4)
      );
    this.prevLoc = this.loc.copy();
    this.v = v || new p5.Vector(p.random(-20, 20), p.random(10, 25));
    this.r = r || p.random(0.5, 1.5);
    this.c = c || getRandomColor(this.p, 128);
  }

  move() {
    this.prevLoc = this.loc.copy();
    const vCopy = this.v.copy();
    vCopy.mult(TIME_STEP);
    this.loc.add(vCopy);
    this.v.mult(1 - (1 - WIND_DRAG) * TIME_STEP);
  }

  applyAcceleration(acceleration: p5.Vector) {
    const accelerationCopy = acceleration.copy();
    accelerationCopy.mult(TIME_STEP);
    this.v.add(accelerationCopy);
  }

  checkEdges() {
    if (this.loc.x <= this.r - REF_W / 2) {
      this.v.x *= -FRICTION;
      this.loc.x = this.r - REF_W / 2 + NUDGE * TIME_STEP;
    }
    if (this.loc.x >= REF_W / 2 - this.r) {
      this.v.x *= -FRICTION;
      this.loc.x = REF_W / 2 - this.r - NUDGE * TIME_STEP;
    }
    if (this.loc.y <= this.r + REF_U) {
      this.v.y *= -FRICTION;
      this.loc.y = this.r + REF_U + NUDGE * TIME_STEP;
      if (this.loc.x <= 0) {
        return -1;
      } else {
        return 1;
      }
    }
    if (this.loc.y >= REF_H - this.r) {
      this.v.y *= -FRICTION;
      this.loc.y = REF_H - this.r - NUDGE * TIME_STEP;
    }
    if (
      this.loc.x <= REF_WALLHEIGHT / 2 + this.r &&
      this.prevLoc.x > REF_WALLWIDTH / 2 + this.r &&
      this.loc.y <= REF_WALLHEIGHT
    ) {
      this.v.x *= -FRICTION;
      this.loc.x = REF_WALLWIDTH / 2 + this.r + NUDGE * TIME_STEP;
    }
    if (
      this.loc.x >= -REF_WALLWIDTH / 2 - this.r &&
      this.prevLoc.x < -REF_WALLWIDTH / 2 - this.r &&
      this.loc.y <= REF_WALLHEIGHT
    ) {
      this.v.x *= -FRICTION;
      this.loc.x = -REF_WALLWIDTH / 2 - this.r - NUDGE * TIME_STEP;
    }
    return 0;
  }

  getDist2(p: Particle) {
    const dy = p.loc.y - this.loc.y;
    const dx = p.loc.x - this.loc.x;
    return dx * dx + dy * dy;
  }

  isColliding(p: Particle) {
    const r = this.r + p.r;
    return r * r > this.getDist2(p);
  }

  bounce(p: Particle) {
    const ab = new p5.Vector();
    ab.set(this.loc);
    ab.sub(p.loc);
    ab.normalize();
    ab.mult(NUDGE);
    while (this.isColliding(p)) {
      this.loc.add(ab);
    }
    const n = p5.Vector.sub(this.loc, p.loc);
    n.normalize();
    const u = p5.Vector.sub(this.v, p.v);
    const un = n.copy();
    un.mult(u.dot(n) * 2);
    // const un = p5.Vector.mult(n, u.dot(n) * 2);
    u.sub(un);
    this.v = p5.Vector.add(u, p.v);
  }

  limitSpeed(minSpeed: number, maxSpeed: number) {
    const mag2 = this.v.magSq();
    if (mag2 > maxSpeed * maxSpeed) {
      this.v.normalize();
      this.v.mult(maxSpeed);
    }
    if (mag2 < minSpeed * minSpeed) {
      this.v.normalize();
      this.v.mult(minSpeed);
    }
  }

  display() {
    this.p.noStroke();
    this.p.fill(this.c);
    this.p.ellipse(
      toX(this.loc.x),
      toY(this.p, this.loc.y),
      toP(this.r) * 2,
      toP(this.r) * 2
    );
  }
}
