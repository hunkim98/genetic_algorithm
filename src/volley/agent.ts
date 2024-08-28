import p5, { Vector } from "p5";
import { Brain } from "./brain";
import {
  INIT_GENE,
  PLAYER_SPEED_X,
  PLAYER_SPEED_Y,
  TIME_STEP,
} from "./constant";
import { Game } from "./game";

export class Agent {
  // dir -1 means left, 1 means right (this is for symmetry)
  dir: number;
  // loc is the location of the agent
  loc: p5.Vector;
  // v is the velocity of the agent
  v: p5.Vector;
  // desiredVelocity is the desired velocity of the agent
  desiredVelocity: p5.Vector;
  // r is the radius of the agent
  r: number;
  // c is the color of the agent
  c: number;
  // opponent is the opponent agent
  opponent: Agent | null;
  // score is the score of the agent
  score: number;
  // this emotion is trivial
  emotion: string;
  // fontSize of the score
  scoreFontSize: number;
  action: {
    // the current set of actions the agent wants to take
    forward: boolean;
    backward: boolean;
    jump: boolean;
  };
  actionIntensity: [number, number, number];
  state: {
    x: number;
    y: number;
    vx: number;
    vy: number;
    bx: number;
    by: number;
    bvx: number;
    bvy: number;
  };
  brain: Brain;
  game: Game;

  constructor(dir: number, loc: p5.Vector, c: number, game: Game) {
    this.dir = dir;
    // 24 * 2 is the width of the game
    // we divide by 4 to get the width of the game in the p5.js coordinate system
    this.loc = loc || new p5.Vector((24 * 2) / 4, 1.5);
    this.v = new p5.Vector(0, 0);
    this.desiredVelocity = new p5.Vector(0, 0);
    this.r = 1.5;
    this.c = c;
    this.opponent = null;
    this.score = 0;
    this.emotion = "happy";
    this.scoreFontSize = 64;
    this.action = {
      forward: false,
      backward: false,
      jump: false,
    };
    this.actionIntensity = [0, 0, 0];
    this.state = {
      x: 0,
      y: 0,
      vx: 0,
      vy: 0,
      bx: 0,
      by: 0,
      bvx: 0,
      bvy: 0,
    };
    this.brain = new Brain({
      initGene: INIT_GENE,
    });
    this.game = game;
  }

  setOpponent(opponent: Agent) {
    this.opponent = opponent;
  }

  setAction(forward: boolean, backward: boolean, jump: boolean) {
    this.action.forward = forward;
    this.action.backward = backward;
    this.action.jump = jump;
  }

  setBrainAction() {
    const forward = this.brain.outputState[0] > 0.75;
    const backward = this.brain.outputState[1] > 0.75;
    const jump = this.brain.outputState[2] > 0.75;
    this.setAction(forward, backward, jump);
  }

  processAction() {
    const forward = this.action.forward;
    const backward = this.action.backward;
    const jump = this.action.jump;
    this.desiredVelocity.x = 0;
    this.desiredVelocity.y = 0;

    if (forward && !backward) {
      this.desiredVelocity.x = -PLAYER_SPEED_X;
    }
    if (backward && !forward) {
      this.desiredVelocity.x = PLAYER_SPEED_X;
    }

    if (jump) {
      this.desiredVelocity.y = PLAYER_SPEED_Y;
    }
  }

  move() {
    const delta = this.v.copy();
    delta.mult(TIME_STEP);
    this.loc.add(delta);
  }

  getState() {
    this.state = {
      x: this.loc.x * this.dir,
      y: this.loc.y,
      vx: this.v.x * this.dir,
      vy: this.v.y,
      bx: this.game.ball.loc.x * this.dir,
      by: this.game.ball.loc.y,
      bvx: this.game.ball.v.x * this.dir,
      bvy: this.game.ball.v.y,
    };
    return this.state;
  }

  printState() {
    let stateText = "";
    const state = this.getState();
    stateText += "X: " + Math.round(state.x * r) / r + "\n";
    stateText += "Y: " + Math.round(state.y * r) / r + "\n";
    stateText += "vx: " + Math.round(state.vx * r) / r + "\n";
    stateText += "vy: " + Math.round(state.vy * r) / r + "\n";
    stateText += "bx: " + Math.round(state.bx * r) / r + "\n";
    stateText += "by: " + Math.round(state.by * r) / r + "\n";
    stateText += "bvx: " + Math.round(state.bvx * r) / r + "\n";
    stateText += "bvy: " + Math.round(state.bvy * r) / r + "\n";
    fill(this.c);
    stroke(this.c);
    textFont("Courier New");
    textSize(16);
    text(stateText, toX((this.dir * ref_w) / 4), toP(ref_u));
  }

  drawState(human: boolean) {
    const brain = this.brain;
    const r = red(this.c);
    const g = green(this.c);
    const b = blue(this.c);
    let i,
      j = 0;
    let temp;
    const radius = ref_w / 2 / (brain.nGameInput - 4 + 4);
    const factor = 3 / 4;
    const startX = ref_w / 4 - radius * ((brain.nGameInput - 4) / 2);
    const ballFactor = 1.0;
    const startX2 = ref_w / 4 - ballFactor * radius * (brain.nGameOutput / 2);
    const secondLayerY = Math.max(
      (height * 1) / 8 + toP(radius) + 0.5 * toP(radius),
      (height * 3) / 16
    );

    this.actionIntensity[0] += this.action.forward ? 16 : 0;
    this.actionIntensity[1] += this.action.jump ? 16 : 0;
    this.actionIntensity[2] += this.action.backward ? 16 : 0;

    if (!human) {
      for (i = 0; i < brain.nGameInput - 4; i++) {
        noStroke();
        fill(r, g, b, brain.inputState[i] * 32 + 8);
        ellipse(
          toX((startX + i * radius) * this.dir),
          (height * 1) / 8 + toP(radius),
          toP(radius * factor),
          toP(radius * factor)
        );

        for (j = 0; j < brain.nGameOutput; j++) {
          if (this.actionIntensity[j] > 64) {
            stroke(r, g, b, brain.inputState[i] * 32);
            line(
              toX((startX + i * radius) * this.dir),
              (height * 1) / 8 + toP(radius),
              toX((startX2 + ballFactor * j * radius) * this.dir),
              secondLayerY + (ballFactor + 0) * toP(radius)
            );
          }
        }
      }
    }

    for (j = 0; j < brain.nGameOutput; j++) {
      this.actionIntensity[j] -= 4;
      this.actionIntensity[j] = Math.min(this.actionIntensity[j], 128);
      this.actionIntensity[j] = Math.max(this.actionIntensity[j], 16);

      noStroke();
      fill(r, g, b, this.actionIntensity[j]);
      ellipse(
        toX((startX2 + ballFactor * j * radius) * this.dir),
        secondLayerY + (ballFactor + 0) * toP(radius),
        toP(radius * factor) * ballFactor,
        toP(radius * factor) * ballFactor
      );
    }
  }

  update() {
    this.v.add(p5.Vector.mult(gravity, timeStep));
    if (this.loc.y <= ref_u + nudge * timeStep) {
      this.v.y = this.desiredVelocity.y;
    }
    this.v.x = this.desiredVelocity.x * this.dir;
    this.move();
    if (this.loc.y <= ref_u) {
      this.loc.y = ref_u;
      this.v.y = 0;
    }

    if (this.loc.x * this.dir <= ref_wallwidth / 2 + this.r) {
      this.v.x = 0;
      this.loc.x = this.dir * (ref_wallwidth / 2 + this.r);
    }
    if (this.loc.x * this.dir >= ref_w / 2 - this.r) {
      this.v.x = 0;
      this.loc.x = this.dir * (ref_w / 2 - this.r);
    }
  }

  display() {
    const x = this.loc.x;
    const y = this.loc.y;
    const r = this.r;
    let angle = 60;
    let eyeX = 0;
    let eyeY = 0;

    if (this.dir === 1) angle = 135;
    noStroke();
    fill(this.c);
    arc(toX(x), toY(y), toP(r) * 2, toP(r) * 2, Math.PI, 2 * Math.PI);

    var ballX = game.ball.loc.x - (x + 0.6 * r * fastCos(angle));
    var ballY = game.ball.loc.y - (y + 0.6 * r * fastSin(angle));
    if (this.emotion === "sad") {
      ballX = -this.dir;
      ballY = -3;
    }
    var dist = Math.sqrt(ballX * ballX + ballY * ballY);
    eyeX = ballX / dist;
    eyeY = ballY / dist;

    fill(255);
    ellipse(
      toX(x + 0.6 * r * fastCos(angle)),
      toY(y + 0.6 * r * fastSin(angle)),
      toP(r) * 0.6,
      toP(r) * 0.6
    );
    fill(0);
    ellipse(
      toX(x + 0.6 * r * fastCos(angle) + eyeX * 0.15 * r),
      toY(y + 0.6 * r * fastSin(angle) + eyeY * 0.15 * r),
      toP(r) * 0.2,
      toP(r) * 0.2
    );
  }

  drawScore() {
    const r = red(this.c);
    const g = green(this.c);
    const b = blue(this.c);
    const size = this.scoreFontSize;
    const factor = 0.95;
    this.scoreFontSize =
      baseScoreFontSize + (this.scoreFontSize - baseScoreFontSize) * factor;

    if (this.score > 0) {
      textFont("Courier New");
      textSize(size);
      stroke(r, g, b, 128 * (baseScoreFontSize / this.scoreFontSize));
      fill(r, g, b, 64 * (baseScoreFontSize / this.scoreFontSize));
      textAlign(this.dir === -1 ? LEFT : RIGHT);
      text(
        this.score,
        this.dir === -1 ? (size * 3) / 4 : width - size / 4,
        size / 2 + height / 3
      );
    }
  }
}
