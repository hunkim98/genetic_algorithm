import p5, { Color, Vector } from "p5";
import { Brain } from "./brain";
import {
  INIT_GENE,
  PLAYER_SPEED_X,
  PLAYER_SPEED_Y,
  REF_H,
  REF_U,
  REF_W,
  THE_GRAVITY,
  TIME_STEP,
  NUDGE,
  REF_WALLWIDTH,
  BASE_SCORE_FONT_SIZE,
} from "./constant";
import { Game } from "./game";
import { fastCos, fastSin, toP, toX, toY } from "./utils";

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
  c: Color;
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
  p: p5;

  constructor(p: p5, dir: number, loc: p5.Vector, c: Color, game: Game) {
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
    this.p = p;
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
    const r = 10;
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
    this.p.fill(this.c);
    this.p.stroke(this.c);
    this.p.textFont("Courier New");
    this.p.textSize(16);
    this.p.text(stateText, toX((this.dir * REF_W) / 4), toP(REF_U));
  }

  drawState(human: boolean) {
    const brain = this.brain;
    const r = this.p.red(this.c);
    const g = this.p.green(this.c);
    const b = this.p.blue(this.c);
    let i,
      j = 0;
    let temp;
    const radius = REF_W / 2 / (brain.nGameInput - 4 + 4);
    const factor = 3 / 4;
    const startX = REF_W / 4 - radius * ((brain.nGameInput - 4) / 2);
    const ballFactor = 1.0;
    const startX2 = REF_W / 4 - ballFactor * radius * (brain.nGameOutput / 2);
    const secondLayerY = Math.max(
      (this.p.height * 1) / 8 + toP(radius) + 0.5 * toP(radius),
      (this.p.height * 3) / 16
    );

    this.actionIntensity[0] += this.action.forward ? 16 : 0;
    this.actionIntensity[1] += this.action.jump ? 16 : 0;
    this.actionIntensity[2] += this.action.backward ? 16 : 0;

    if (!human) {
      for (i = 0; i < brain.nGameInput - 4; i++) {
        this.p.noStroke();
        this.p.fill(r, g, b, brain.inputState[i] * 32 + 8);
        this.p.ellipse(
          toX((startX + i * radius) * this.dir),
          (this.p.height * 1) / 8 + toP(radius),
          toP(radius * factor),
          toP(radius * factor)
        );

        for (j = 0; j < brain.nGameOutput; j++) {
          if (this.actionIntensity[j] > 64) {
            this.p.stroke(r, g, b, brain.inputState[i] * 32);
            this.p.line(
              toX((startX + i * radius) * this.dir),
              (this.p.height * 1) / 8 + toP(radius),
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

      this.p.noStroke();
      this.p.fill(r, g, b, this.actionIntensity[j]);
      this.p.ellipse(
        toX((startX2 + ballFactor * j * radius) * this.dir),
        secondLayerY + (ballFactor + 0) * toP(radius),
        toP(radius * factor) * ballFactor,
        toP(radius * factor) * ballFactor
      );
    }
  }

  update() {
    this.v.add(p5.Vector.mult(THE_GRAVITY, TIME_STEP));
    if (this.loc.y <= REF_U + NUDGE * TIME_STEP) {
      this.v.y = this.desiredVelocity.y;
    }
    this.v.x = this.desiredVelocity.x * this.dir;
    this.move();
    if (this.loc.y <= REF_U) {
      this.loc.y = REF_U;
      this.v.y = 0;
    }

    if (this.loc.x * this.dir <= REF_WALLWIDTH / 2 + this.r) {
      this.v.x = 0;
      this.loc.x = this.dir * (REF_WALLWIDTH / 2 + this.r);
    }
    if (this.loc.x * this.dir >= REF_W / 2 - this.r) {
      this.v.x = 0;
      this.loc.x = this.dir * (REF_W / 2 - this.r);
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
    this.p.noStroke();
    this.p.fill(this.c);
    this.p.arc(
      toX(x),
      toY(this.p, y),
      toP(r) * 2,
      toP(r) * 2,
      Math.PI,
      2 * Math.PI
    );

    var ballX = this.game.ball.loc.x - (x + 0.6 * r * fastCos(angle));
    var ballY = this.game.ball.loc.y - (y + 0.6 * r * fastSin(angle));
    if (this.emotion === "sad") {
      ballX = -this.dir;
      ballY = -3;
    }
    var dist = Math.sqrt(ballX * ballX + ballY * ballY);
    eyeX = ballX / dist;
    eyeY = ballY / dist;

    this.p.fill(255);
    this.p.ellipse(
      toX(x + 0.6 * r * fastCos(angle)),
      toY(this.p, y + 0.6 * r * fastSin(angle)),
      toP(r) * 0.6,
      toP(r) * 0.6
    );
    this.p.fill(0);
    this.p.ellipse(
      toX(x + 0.6 * r * fastCos(angle) + eyeX * 0.15 * r),
      toY(this.p, y + 0.6 * r * fastSin(angle) + eyeY * 0.15 * r),
      toP(r) * 0.2,
      toP(r) * 0.2
    );
  }

  drawScore() {
    const r = this.p.red(this.c);
    const g = this.p.green(this.c);
    const b = this.p.blue(this.c);
    const size = this.scoreFontSize;
    const factor = 0.95;
    this.scoreFontSize =
      BASE_SCORE_FONT_SIZE +
      (this.scoreFontSize - BASE_SCORE_FONT_SIZE) * factor;

    if (this.score > 0) {
      this.p.textFont("Courier New");
      this.p.textSize(size);
      this.p.stroke(r, g, b, 128 * (BASE_SCORE_FONT_SIZE / this.scoreFontSize));
      this.p.fill(r, g, b, 64 * (BASE_SCORE_FONT_SIZE / this.scoreFontSize));
      this.p.textAlign(this.dir === -1 ? this.p.LEFT : this.p.RIGHT);
      this.p.text(
        this.score,
        this.dir === -1 ? (size * 3) / 4 : this.p.width - size / 4,
        size / 2 + this.p.height / 3
      );
    }
  }
}
