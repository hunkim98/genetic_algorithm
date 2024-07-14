import { zeros } from "../array";
import { randi } from "../random";
import { Chromosome, getNetworkSize, pushGeneToNetwork } from "./chromosome";

class GATrainer {
  private net: any;
  private population_size: number;
  private mutation_rate: number;
  private elite_percentage: number;
  private mutation_size: number;
  private target_fitness: number;
  private burst_generations: number;
  private best_trial: number;
  private num_match: number;
  private chromosome_size: number;
  private chromosomes: Chromosome[];
  private bestFitness: number;
  private bestFitnessCount: number;

  constructor(net: any, options_?: any, initGene?: number[]) {
    this.net = net;

    const options = options_ || {};
    this.population_size =
      options.population_size !== undefined ? options.population_size : 100;
    this.population_size = Math.floor(this.population_size / 2) * 2; // make sure even number
    this.mutation_rate =
      options.mutation_rate !== undefined ? options.mutation_rate : 0.01;
    this.elite_percentage =
      options.elite_percentage !== undefined ? options.elite_percentage : 0.2;
    this.mutation_size =
      options.mutation_size !== undefined ? options.mutation_size : 0.05;
    this.target_fitness =
      options.target_fitness !== undefined
        ? options.target_fitness
        : 10000000000000000;
    this.burst_generations =
      options.burst_generations !== undefined ? options.burst_generations : 10;
    this.best_trial = options.best_trial !== undefined ? options.best_trial : 1;
    this.num_match = options.num_match !== undefined ? options.num_match : 1;
    this.chromosome_size = getNetworkSize(this.net);

    let initChromosome: Chromosome | null = null;
    if (initGene) {
      initChromosome = new Chromosome(initGene);
    }

    this.chromosomes = []; // population
    for (let i = 0; i < this.population_size; i++) {
      const chromosome = new Chromosome(zeros(this.chromosome_size));
      if (initChromosome) {
        // if initial gene supplied, burst mutate param.
        chromosome.copyFrom(initChromosome);
        if (i > 0) {
          // don't mutate the first guy.
          chromosome.burst_mutate(this.mutation_size);
        }
      } else {
        chromosome.randomize(1.0);
      }
      this.chromosomes.push(chromosome);
    }
    pushGeneToNetwork(this.net, this.chromosomes[0].gene); // push first chromosome to neural network.

    this.bestFitness = -10000000000000000;
    this.bestFitnessCount = 0;
  }

  train(fitFunc: (net: any) => number): number {
    const bestFitFunc = (nTrial: number, net: any): number => {
      let bestFitness = -10000000000000000;
      let fitness: number;
      for (let i = 0; i < nTrial; i++) {
        fitness = fitFunc(net);
        if (fitness > bestFitness) {
          bestFitness = fitness;
        }
      }
      return bestFitness;
    };

    let i: number, N: number;
    let fitness: number;
    let c = this.chromosomes;
    N = this.population_size;

    let bestFitness = -10000000000000000;

    // process first net (the best one)
    pushGeneToNetwork(this.net, c[0].gene);
    fitness = bestFitFunc(this.best_trial, this.net);
    c[0].fitness = fitness;
    bestFitness = fitness;
    if (bestFitness > this.target_fitness) {
      return bestFitness;
    }

    for (i = 1; i < N; i++) {
      pushGeneToNetwork(this.net, c[i].gene);
      fitness = bestFitFunc(this.best_trial, this.net);
      c[i].fitness = fitness;
      if (fitness > bestFitness) {
        bestFitness = fitness;
      }
    }

    // sort the chromosomes by fitness
    c = c.sort((a, b) => {
      if (a.fitness > b.fitness) {
        return -1;
      }
      if (a.fitness < b.fitness) {
        return 1;
      }
      return 0;
    });

    const Nelite = Math.floor(Math.floor(this.elite_percentage * N) / 2) * 2; // even number
    for (i = Nelite; i < N; i += 2) {
      const p1 = randi(0, Nelite);
      const p2 = randi(0, Nelite);
      c[p1].crossover(c[p2], c[i], c[i + 1]);
    }

    for (i = 1; i < N; i++) {
      // keep best guy the same.  don't mutate the best one, so start from 1, not 0.
      c[i].mutate(this.mutation_rate, this.mutation_size);
    }

    // push best one to network.
    pushGeneToNetwork(this.net, c[0].gene);
    if (bestFitness < this.bestFitness) {
      // didn't beat the record this time
      this.bestFitnessCount++;
      if (this.bestFitnessCount > this.burst_generations) {
        // stagnation, do burst mutate!
        for (i = 1; i < N; i++) {
          c[i].copyFrom(c[0]);
          c[i].burst_mutate(this.mutation_size);
        }
      }
    } else {
      this.bestFitnessCount = 0; // reset count for burst
      this.bestFitness = bestFitness; // record the best fitness score
    }

    return bestFitness;
  }

  matchTrain(
    matchFunc: (chromosome1: Chromosome, chromosome2: Chromosome) => number
  ): void {
    let i: number, j: number, N: number;
    let opponent: number;
    let fitness: number;
    let c = this.chromosomes;
    let result = 0;
    N = this.population_size;

    // zero out all fitness and
    for (i = 0; i < N; i++) {
      c[i].fitness = 0;
      c[i].nTrial = 0;
    }

    // get these guys to fight against each other!
    for (i = 0; i < N; i++) {
      for (j = 0; j < this.num_match; j++) {
        opponent = randi(0, N);
        if (opponent === i) continue;
        result = matchFunc(c[i], c[opponent]);
        c[i].nTrial += 1;
        c[opponent].nTrial += 1;
        c[i].fitness += result + 1;
        c[opponent].fitness += -result + 1; // if result is -1, it means opponent has won.
      }
    }

    // average out all fitness scores by the number of matches each chromosome has done.
    for (i = 0; i < N; i++) {
      if (c[i].nTrial > 0) {
        c[i].fitness /= c[i].nTrial;
      }
    }

    // sort the chromosomes by fitness
    c = c.sort((a, b) => {
      if (a.fitness > b.fitness) {
        return -1;
      }
      if (a.fitness < b.fitness) {
        return 1;
      }
      return 0;
    });

    const Nelite = Math.floor(Math.floor(this.elite_percentage * N) / 2) * 2; // even number
    for (i = Nelite; i < N; i += 2) {
      const p1 = randi(0, Nelite);
      const p2 = randi(0, Nelite);
      c[p1].crossover(c[p2], c[i], c[i + 1]);
    }

    for (i = 2; i < N; i++) {
      // keep two best guys the same.  don't mutate the best one, so start from 2, not 0.
      c[i].mutate(this.mutation_rate, this.mutation_size);
    }

    // push best one to network.
    pushGeneToNetwork(this.net, c[0].gene);
  }
}
