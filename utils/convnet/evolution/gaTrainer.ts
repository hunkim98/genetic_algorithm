import { zeros } from "../array";
import { Net } from "../net";
import { randi } from "../random";
import { Chromosome, getNetworkSize, pushGeneToNetwork } from "./chromosome";

export interface GATrainerOptions {
  population_size?: number;
  mutation_rate?: number;
  elite_percentage?: number;
  mutation_size?: number;
  target_fitness?: number;
  burst_generations?: number;
  best_trial?: number;
  num_match?: number;
}

export class GATrainer {
  private net: Net;
  private population_size: number;
  private mutation_rate: number;
  private elite_percentage: number;
  private mutation_size: number;
  private target_fitness: number;
  private burst_generations: number;
  private best_trial: number;
  private num_match: number;
  private chromosome_size: number;
  public chromosomes: Chromosome[];
  private bestFitness: number;
  private bestFitnessCount: number;

  constructor(
    net: Net,
    options_?: GATrainerOptions,
    initGene?: number[] | Float64Array
  ) {
    this.net = net;

    const options = options_ || {};
    // population size is the number of chromosomes in the population.
    this.population_size =
      options.population_size !== undefined ? options.population_size : 100;
    // we need even number of population size for crossover
    this.population_size = Math.floor(this.population_size / 2) * 2; // make sure even number
    // mutation rate is the probability of mutation.
    this.mutation_rate =
      options.mutation_rate !== undefined ? options.mutation_rate : 0.01;
    // elite percentage is the percentage of the population that will be kept as is.
    // they will be the parent of the next generation.
    this.elite_percentage =
      options.elite_percentage !== undefined ? options.elite_percentage : 0.2;
    // mutation size is the standard deviation of the gaussian noise added to the gene.
    this.mutation_size =
      options.mutation_size !== undefined ? options.mutation_size : 0.05;
    // target fitness is the fitness score that we want to achieve.
    this.target_fitness =
      options.target_fitness !== undefined
        ? options.target_fitness
        : 10000000000000000;
    // burst generations is the number of generations that the best fitness score has not improved.
    this.burst_generations =
      options.burst_generations !== undefined ? options.burst_generations : 10;
    // best trial is the number of trials to get the best fitness score.
    this.best_trial = options.best_trial !== undefined ? options.best_trial : 1;
    // num match is the number of matches each chromosome will have.
    this.num_match = options.num_match !== undefined ? options.num_match : 1;
    // chromosome size is the number of genes in the chromosome.
    // it is the number of weights and biases in the neural network.
    this.chromosome_size = getNetworkSize(this.net);

    console.log("chromosome size: " + this.chromosome_size);

    let initChromosome: Chromosome | null = null;
    // initGene is for the initial gene of the first chromosome.
    if (initGene) {
      initChromosome = new Chromosome(initGene);
    }

    // create population
    this.chromosomes = []; // population
    for (let i = 0; i < this.population_size; i++) {
      const chromosome = new Chromosome(zeros(this.chromosome_size));
      if (initChromosome) {
        // if initial gene supplied, burst mutate param.
        // burst means that we will copy the first chromosome and mutate it.
        chromosome.copyFrom(initChromosome);
        if (i > 0) {
          // don't mutate the first guy.
          // the first guy should not be mutated.
          chromosome.burst_mutate(this.mutation_size);
        }
      } else {
        // if there is no initial gene, randomize the gene.
        // 1.0 is the standard deviation of the gaussian noise.
        chromosome.randomize(1.0);
      }
      this.chromosomes.push(chromosome);
    }
    pushGeneToNetwork(this.net, this.chromosomes[0].gene); // push first chromosome to neural network.

    this.bestFitness = -10000000000000000;
    this.bestFitnessCount = 0;
  }

  // the fitFunc is the function that will be used to evaluate the fitness of the chromosome.
  // this is like the objective function in optimization.
  train(fitFunc: (net: Net) => number): number {
    // we first create a function that will get the best fitness score of a chromosome.
    const bestFitFunc = (nTrial: number, net: Net): number => {
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
    // N is the population size.
    N = this.population_size;

    let bestFitness = -10000000000000000;

    // process first net (the best one)
    pushGeneToNetwork(this.net, c[0].gene);
    // we give the chromosome a limit of best_trial to get the best fitness score.
    fitness = bestFitFunc(this.best_trial, this.net);
    // we set the fitness score of the chromosome.
    // each chromosome object has a fitness property.
    c[0].fitness = fitness;
    // since this is the first chromosome, we set the best fitness score to the fitness score of the first chromosome.
    bestFitness = fitness;
    // if the best fitness score is already greater than the target fitness score,
    // we do not need to train other chromosomes.
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
    // the best chromosome will be the first one.
    c = c.sort((a, b) => {
      if (a.fitness > b.fitness) {
        // a is put before b if a has greater fitness score.
        return -1;
      }
      if (a.fitness < b.fitness) {
        return 1;
      }
      return 0;
    });

    // Nelite is the number of elite chromosomes.
    // if the elite percentage is 0.2 and the population size is 100,
    // then Nelite will be 20.
    const Nelite = Math.floor(Math.floor(this.elite_percentage * N) / 2) * 2; // even number
    for (i = Nelite; i < N; i += 2) {
      // since we have sorted the chromosomes by fitness,
      // the chromosomes with an index from 0 to Nelite - 1 are the elite chromosomes.
      const p1 = randi(0, Nelite);
      const p2 = randi(0, Nelite);
      // the parent chromosomes will be p1 and p2.
      c[p1].crossover(c[p2], c[i], c[i + 1]);
    }

    // In genetic algorithm, we mutate the genes of the chromosomes.
    for (i = 1; i < N; i++) {
      // c[0] will be the one with the best fitness score.
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

  // this is a function for training two agents (= chromosomes) against each other.
  // uses arms race to determine best chromosome by playing them against each other
  // this algorithm loops through each chromosome, and for each chromosome, it will play num_match games
  // against other chromosomes.  at the same time.  if it wins, the fitness is incremented by 1
  // else it is subtracted by 1.  if the game is tied, the fitness doesn't change.
  // at the end of the algorithm, each fitness is divided by the number of games the chromosome has played
  // the algorithm will then sort the chromosomes by this average fitness
  matchTrain(
    matchFunc: (chromosome1: Chromosome, chromosome2: Chromosome) => number
  ): void {
    let i: number, j: number, N: number;
    let opponent: number;
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
        // choose a random opponent
        opponent = randi(0, N);
        // if the opponent is myself, skip
        if (opponent === i) continue;
        // use the matchFunc passed as the argument to determine the result of the match
        // the matchFunc should shw how the two chromosomes will compete against each other.
        // the result will be either 1, 0, or -1.
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
    // same as above train function.
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
