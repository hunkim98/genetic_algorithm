export class DNA {
  genes: string[];
  fitness: number;

  constructor({ length }: { length: number }) {
    this.genes = [];
    this.fitness = 0;
    for (let i = 0; i < length; i++) {
      this.genes.push(DNA.randomGene());
    }
  }

  static randomGene() {
    const maxCharacterCode = 122;
    const minCharacterCode = 63;
    let c =
      Math.floor(Math.random() * (maxCharacterCode - minCharacterCode)) +
      minCharacterCode;
    if (c === 63) c = 32;
    if (c === 64) c = 46;
    return String.fromCharCode(c);
  }

  getPhrase() {
    return this.genes.join("");
  }

  calcFitness(target: string) {
    let score = 0;
    for (let i = 0; i < this.genes.length; i++) {
      if (this.genes[i] === target[i]) {
        score++;
      }
    }
    // normalize the score
    this.fitness = score / target.length;
  }

  crossover(partner: DNA) {
    const child = new DNA({ length: this.genes.length });
    // random midpoint
    const midpoint = Math.floor(Math.random() * this.genes.length);
    for (let i = 0; i < this.genes.length; i++) {
      if (i > midpoint) {
        child.genes[i] = this.genes[i];
      } else {
        child.genes[i] = partner.genes[i];
      }
    }
    return child;
  }

  mutate(mutationRate: number) {
    for (let i = 0; i < this.genes.length; i++) {
      if (Math.random() < mutationRate) {
        this.genes[i] = DNA.randomGene();
      }
    }
  }
}
