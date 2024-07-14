import { zeros } from "../array";
import { randf, randi, randn } from "../random";

class Chromosome {
  fitness: number = 0; // default value
  nTrial: number = 0; // number of trials subjected to so far.
  gene: number[];

  constructor(floatArray: number[]) {
    this.gene = floatArray;
  }

  burst_mutate(burst_magnitude_: number): void {
    // adds a normal random variable of stdev width, zero mean to each gene.
    const burst_magnitude: number = burst_magnitude_ || 0.1;
    const N: number = this.gene.length;
    for (let i = 0; i < N; i++) {
      this.gene[i] += randn(0.0, burst_magnitude);
    }
  }

  randomize(burst_magnitude_: number): void {
    // resets each gene to a random value with zero mean and stdev
    const burst_magnitude: number = burst_magnitude_ || 0.1;
    const N: number = this.gene.length;
    for (let i = 0; i < N; i++) {
      this.gene[i] = randn(0.0, burst_magnitude);
    }
  }

  mutate(mutation_rate_: number, burst_magnitude_: number): void {
    // adds random gaussian (0,stdev) to each gene with prob mutation_rate
    const mutation_rate: number = mutation_rate_ || 0.1;
    const burst_magnitude: number = burst_magnitude_ || 0.1;
    const N: number = this.gene.length;
    for (let i = 0; i < N; i++) {
      if (randf(0, 1) < mutation_rate) {
        this.gene[i] += randn(0.0, burst_magnitude);
      }
    }
  }

  crossover(partner: Chromosome, kid1: Chromosome, kid2: Chromosome): void {
    // performs one-point crossover with partner to produce 2 kids
    // assumes all chromosomes are initialised with same array size. pls make sure of this before calling
    const N: number = this.gene.length;
    const l: number = randi(0, N); // crossover point
    for (let i = 0; i < N; i++) {
      if (i < l) {
        kid1.gene[i] = this.gene[i];
        kid2.gene[i] = partner.gene[i];
      } else {
        kid1.gene[i] = partner.gene[i];
        kid2.gene[i] = this.gene[i];
      }
    }
  }

  copyFrom(c: Chromosome): void {
    // copies c's gene into itself
    this.copyFromGene(c.gene);
  }

  copyFromGene(gene: number[]): void {
    // gene into itself
    const N: number = this.gene.length;
    for (let i = 0; i < N; i++) {
      this.gene[i] = gene[i];
    }
  }

  clone(): Chromosome {
    // returns an exact copy of itself (into new memory, doesn't return reference)
    const newGene: number[] = zeros(this.gene.length);
    for (let i = 0; i < this.gene.length; i++) {
      newGene[i] = Math.round(10000 * this.gene[i]) / 10000;
    }
    const c: Chromosome = new Chromosome(newGene);
    c.fitness = this.fitness;
    return c;
  }

  pushToNetwork(net: any): void {
    // pushes this chromosome to a specified network
    pushGeneToNetwork(net, this.gene);
  }
}

function pushGeneToNetwork(net, gene) {
  // pushes the gene (floatArray) to fill up weights and biases in net
  var count = 0;
  var layer = null;
  var filter = null;
  var bias = null;
  var w = null;
  var i, j, k;
  for (i = 0; i < net.layers.length; i++) {
    layer = net.layers[i];
    filter = layer.filters;
    if (filter) {
      for (j = 0; j < filter.length; j++) {
        w = filter[j].w;
        for (k = 0; k < w.length; k++) {
          w[k] = gene[count++];
        }
      }
    }
    bias = layer.biases;
    if (bias) {
      w = bias.w;
      for (k = 0; k < w.length; k++) {
        w[k] = gene[count++];
      }
    }
  }
}

function getGeneFromNetwork(net) {
  // gets all the weight/biases from network in a floatArray
  var gene = [];
  var layer = null;
  var filter = null;
  var bias = null;
  var w = null;
  var i, j, k;
  for (i = 0; i < net.layers.length; i++) {
    layer = net.layers[i];
    filter = layer.filters;
    if (filter) {
      for (j = 0; j < filter.length; j++) {
        w = filter[j].w;
        for (k = 0; k < w.length; k++) {
          gene.push(w[k]);
        }
      }
    }
    bias = layer.biases;
    if (bias) {
      w = bias.w;
      for (k = 0; k < w.length; k++) {
        gene.push(w[k]);
      }
    }
  }
  return gene;
}