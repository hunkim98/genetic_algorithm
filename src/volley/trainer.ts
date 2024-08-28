import { Net } from "@/utils/convnet/net";
import { GATrainer } from "@/utils/convnet/evolution/gatrainer";
import { Brain } from "./brain";
import { Chromosome } from "@/utils/convnet/evolution/chromosome";

class Trainer {
  private net: Net;
  private trainer: GATrainer;
  private matchFunction: (
    chromosome1: Chromosome,
    chromosome2: Chromosome
  ) => number;

  constructor(
    brain: Brain,
    initialGene: number[] | Float64Array,
    matchFunction: (chromosome1: Chromosome, chromosome2: Chromosome) => number
  ) {
    // trainer for neural network interface.  must pass in an initial brain so it knows the net topology.
    // the constructor won't modify the brain object passed in.
    this.net = new Net();
    this.net.makeLayers(brain.layer_defs);

    this.trainer = new GATrainer(
      this.net,
      {
        population_size: 100 * 1,
        mutation_size: 0.3,
        mutation_rate: 0.05,
        num_match: 4 * 2,
        elite_percentage: 0.2,
      },
      initialGene
    );
    this.matchFunction = matchFunction;
  }

  public train(): void {
    console.log("train!");
    this.trainer.matchTrain(this.matchFunction);
  }

  public getChromosome(n?: number): any {
    // returns a copy of the nth best chromosome (if not provided, returns first one, which is the best one)
    n = n || 0;
    console.log(
      "chromosome " + n + " fitness: " + this.trainer.chromosomes[n].fitness
    );
    return this.trainer.chromosomes[n].clone();
  }
}
