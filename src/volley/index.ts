import { THE_FRAME_RATE, THE_GRAVITY } from "./constant";
import { Game } from "./game";
import "./style.css";
import p5 from "p5";

const canvasWidth = 800;
const canvasHeight = 800;

let myp5 = new p5((sketch: p5) => {
  let gravity = sketch.createVector(0, THE_GRAVITY);
  let game: Game;

  sketch.setup = () => {
    sketch.createCanvas(canvasWidth, canvasHeight);
    sketch.frameRate(THE_FRAME_RATE);
    sketch.background(255);

    // game = new Game
    console.log("hey");
  };

  sketch.draw = () => {};
}, document.getElementById("canvas")!);
