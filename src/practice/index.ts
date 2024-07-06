import p5 from "p5";
import { Population } from "./population";

// For genetic algorithm
// There must be
// 1. Heredity
// 2. Variation
// 3. Selection

// How to code
// 1) Create a "random" population of N elements (setup function) - this is the heredity part (initializa
// 2-1) Calculate fitness for N elements (draw function) - this is the selection part
//      - One of the crucial parts of the genetic algorithm is the fitness function (This should be well defined for the problem
//      - Example) Fitness function for matching a string "HELLO WORLD" would be the number of characters that match the target string
// 2-2) Create a new population by selecting the best elements from the current population (draw function) - this is the selection part
//      - We conduct the selection step for N times (N is the size of the population)
//      - We pick two (this does not always need to be three) parents from the population and create a child by combining the genes of the parents
//      - We can make a new element through either 1) crossover or 2) mutation

// Crossover (=Heridity)
// ex) target: "unicorn" parents: "unijorm" and "popcorn"
// 1) Select a midpoint (or random point) in the parent strings
// 2) Combine the parent strings at the random point "uni | jorm" and "pop | corn"
// 3) The child string is "uni | corn"

// Mutation (=Variation)
// we can mutate the child string by changing one of the characters (We can say that there many be 1% chance of mutation)

// population is an array of DNA objects (N elements), each DNA object has a string and a fitness value
// one element has an array of DNA objects

const canvasWidth = 800;
const canvasHeight = 800;

let myp5 = new p5((sketch: p5) => {
  let x = 0;
  let y = 0;
  let target;
  let popmax: number;
  let mutationRate: number;
  let population: Population;

  let bestPhrase: p5.Element;
  let allPhrases: p5.Element;
  let stats: p5.Element;

  sketch.setup = () => {
    bestPhrase = sketch.createP("Best phrase:");
    bestPhrase.class("best");

    allPhrases = sketch.createP("All phrases:");
    allPhrases.position(600, 10);
    allPhrases.class("all");

    stats = sketch.createP("Stats");
    stats.class("stats");

    target = "To be or not to be.";
    popmax = 500;
    mutationRate = 0.01;

    population = new Population({ p: target, m: mutationRate, num: popmax });
  };

  sketch.draw = () => {
    population.naturalSelection();
    population.generate();
    population.calcFitness();
    population.evaluate();

    if (population.isFinished()) {
      // stop looping
      sketch.noLoop();
    }
    displayInfo();
  };

  const displayInfo = () => {
    let answer = population.getBest();
    bestPhrase.html("Best phrase:<br>" + answer);

    let statstext =
      "total generations:     " + population.getGenerations() + "<br>";
    statstext +=
      "average fitness:       " + population.getAverageFitness() + "<br>";
    statstext += "total population:      " + popmax + "<br>";
    statstext += "mutation rate:         " + mutationRate + "<br>";
    stats.html(statstext);

    allPhrases.html("All phrases:<br>" + population.allPhrases());
  };
}, document.getElementById("canvas")!);
