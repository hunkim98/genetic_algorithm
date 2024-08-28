import p5 from "p5";
import { Particle } from "./particle";

export class Game {
  ball: Particle;
  deadball: any;
  ground: any;
  fence: any;
  fenceStub: any;
  agent1: any;
  agent2: any;
  p: p5;

  constructor(sketch: (p: p5) => void) {
    // this.ball = {
    //   loc: new p5.Vector(0, 0),
    //   v: new p5.Vector(0, 0),
    //   r: 0,
    //   c: 0,
    //   game: this,
    // };
    this.p = new p5(sketch);
    this.ball = new Particle(this.p);
  }
  // this.deadball = new DeadBall();
  // this.ground = new Ground();
  // this.fence = new Fence();
  // this.fenceStub = new FenceStub();
  // this.agent1 = new Agent(-1, new p5.Vector((24 * 2) / 4, 1.5), 0, this);
  // this.agent2 = new Agent(1, new p5.Vector((24 * 2) / 4, 1.5), 1, this);
}
