import { zeros } from "../array";
import { BaseLayer } from "../layer/BaseLayer";
import { Net } from "../net";
import { randf, randi, randn } from "../random";
import { Vol } from "../vol";

export class Chromosome {
  fitness: number = 0; // default value
  nTrial: number = 0; // number of trials subjected to so far.
  gene: number[] | Float64Array;

  constructor(floatArray: number[] | Float64Array) {
    this.gene = floatArray;
  }

  // burst mutate is a special function that adds a random variable to each gene
  // it is different from randomize in that it adds a random variable to the gene
  // while randomize resets the gene to a random value
  burst_mutate(burst_magnitude_: number): void {
    // adds a normal random variable of stdev width, zero mean to each gene.
    const burst_magnitude: number = burst_magnitude_ || 0.1;
    const N: number = this.gene.length;
    for (let i = 0; i < N; i++) {
      // you can see that the gene is mutated by adding a random variable to it
      // this means that the mean of the gene is not zero but actually the value of the original gene
      this.gene[i] += randn(0.0, burst_magnitude);
    }
  }

  randomize(burst_magnitude_: number): void {
    // resets each gene to a random value with zero mean and stdev
    const burst_magnitude: number = burst_magnitude_ || 0.1;
    const N: number = this.gene.length;
    for (let i = 0; i < N; i++) {
      // we are setting the gene to a random value
      this.gene[i] = randn(0.0, burst_magnitude);
    }
  }

  // the difference of mutate to burst_mutate is that mutate has a probability of mutation_rate
  // this means that not all genes will be mutated
  // while burst_mutate mutates all genes
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

  // the element in the gene array is actually the weight/bias of the network
  copyFromGene(gene: number[] | Float64Array): void {
    // gene into itself
    const N: number = this.gene.length;
    for (let i = 0; i < N; i++) {
      this.gene[i] = gene[i];
    }
  }

  clone(): Chromosome {
    // returns an exact copy of itself (into new memory, doesn't return reference)
    const newGene: number[] | Float64Array = zeros(this.gene.length);
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

// counts the number of weights and biases in the network
// this will be used to create a chromosome of the right size
export function getNetworkSize(net: Net) {
  let count = 0;
  let layer: BaseLayer;
  let filter: Array<Vol>;
  let bias: Vol | undefined;
  let w: number[] | Float64Array;
  let i: number, j: number, k: number;

  for (i = 0; i < net.layers.length; i++) {
    layer = net.layers[i];
    if (!layer.filters) continue;
    filter = layer.filters;
    if (filter) {
      for (j = 0; j < filter.length; j++) {
        w = filter[j].w;
        count += w.length;
      }
    }
    bias = layer.biases;
    if (bias) {
      w = bias.w;
      count += w.length;
    }
  }
  return count;
}

// pushGeneToNetwork pushes the gene (floatArray) to fill up weights and biases in net
// this means that the gene elements are the weights and biases of the network
export function pushGeneToNetwork(net: Net, gene: number[] | Float64Array) {
  // pushes the gene (floatArray) to fill up weights and biases in net
  var count = 0;
  let layer: BaseLayer;
  let filter: Array<Vol>;
  let bias: Vol | undefined;
  let w: number[] | Float64Array;
  let i: number, j: number, k: number;
  for (i = 0; i < net.layers.length; i++) {
    layer = net.layers[i];
    // if the layer has no filters, then it is not a convolutional layer
    if (!layer.filters) continue;
    filter = layer.filters;
    if (filter) {
      for (j = 0; j < filter.length; j++) {
        w = filter[j].w;
        for (k = 0; k < w.length; k++) {
          w[k] = gene[count++];
        }
      }
    }
    if (!layer.biases) continue;
    bias = layer.biases;
    if (bias) {
      w = bias.w;
      for (k = 0; k < w.length; k++) {
        w[k] = gene[count++];
      }
    }
  }
}

export function getGeneFromNetwork(net: Net) {
  // gets all the weight/biases from network in a floatArray
  var gene: number[] | Float64Array = [];
  var count = 0;
  let layer: BaseLayer;
  let filter: Array<Vol>;
  let bias: Vol | undefined;
  let w: number[] | Float64Array;
  let i: number, j: number, k: number;
  for (i = 0; i < net.layers.length; i++) {
    layer = net.layers[i];
    if (!layer.filters) continue;
    filter = layer.filters;
    if (filter) {
      for (j = 0; j < filter.length; j++) {
        w = filter[j].w;
        for (k = 0; k < w.length; k++) {
          gene.push(w[k]);
        }
      }
    }
    if (!layer.biases) continue;
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
