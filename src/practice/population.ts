import { DNA } from "./dna";

export class Population {
  population: Array<DNA>;
  matingPool: Array<DNA>;
  generations: number;
  finished: boolean;
  target: string;
  mutationRate: number;
  perfectScore: number;
  best: string;

  constructor({ p, m, num }: { p: string; m: number; num: number }) {
    this.population = [];
    this.matingPool = [];
    this.generations = 0;
    this.finished = false;
    this.target = p;

    this.best = "";
    this.perfectScore = 1;
    this.mutationRate = m;

    for (let i = 0; i < num; i++) {
      this.population.push(new DNA({ length: this.target.length }));
    }

    this.calcFitness();
  }

  calcFitness() {
    for (let i = 0; i < this.population.length; i++) {
      this.population[i].calcFitness(this.target);
    }
  }

  naturalSelection() {
    // clear the mating pool
    this.matingPool = [];

    let maxFitness = 0;
    for (let i = 0; i < this.population.length; i++) {
      if (this.population[i].fitness > maxFitness) {
        maxFitness = this.population[i].fitness;
      }
    }

    for (let i = 0; i < this.population.length; i++) {
      let fitness = this.population[i].fitness / maxFitness;
      // If fitness is higher, then it's more likely to be picked
      // This is called the roulette wheel selection
      let n = Math.floor(fitness * 100);
      for (let j = 0; j < n; j++) {
        this.matingPool.push(this.population[i]);
      }
    }
  }

  generate() {
    for (let i = 0; i < this.population.length; i++) {
      // we choose two random parents
      let a = Math.floor(Math.random() * this.matingPool.length);
      let b = Math.floor(Math.random() * this.matingPool.length);
      let partnerA = this.matingPool[a];
      let partnerB = this.matingPool[b];
      let child = partnerA.crossover(partnerB);
      child.mutate(this.mutationRate);
      this.population[i] = child;
    }
    this.generations++;
  }

  getBest() {
    return this.best;
  }

  evaluate() {
    let worldrecord = 0.0;
    let index = 0;
    for (let i = 0; i < this.population.length; i++) {
      if (this.population[i].fitness > worldrecord) {
        index = i;
        worldrecord = this.population[i].fitness;
      }
    }

    this.best = this.population[index].getPhrase();
    if (worldrecord === this.perfectScore) {
      this.finished = true;
    }
  }

  isFinished() {
    return this.finished;
  }

  getGenerations() {
    return this.generations;
  }

  getAverageFitness() {
    let total = 0;
    for (let i = 0; i < this.population.length; i++) {
      total += this.population[i].fitness;
    }
    return total / this.population.length;
  }

  allPhrases() {
    let everything = "";
    let displayLimit = Math.min(this.population.length, 50);

    for (let i = 0; i < displayLimit; i++) {
      everything += this.population[i].getPhrase() + "<br>";
    }
    return everything;
  }
}
