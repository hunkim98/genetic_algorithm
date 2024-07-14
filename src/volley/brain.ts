import { zeros } from "@/utils/convnet/array";
import { Vol } from "@/utils/convnet/vol";
import { Net } from "@/utils/convnet/net";

// design agent's brain using neural network
class Brain {
  nGameInput: number;
  nGameOutput: number;
  nRecurrentState: number;
  nOutput: number;
  nInput: number;
  inputState: number[] | Float64Array;
  convInputState: any; // Replace with the appropriate type
  outputState: number[] | Float64Array;
  prevOutputState: number[] | Float64Array;
  layer_defs: any[]; // Replace with the appropriate type
  net: any; // Replace with the appropriate type

  constructor() {
    "use strict";
    this.nGameInput = 12; // 8 states for agent, plus 4 state for opponent
    this.nGameOutput = 3; // 3 buttons (forward, backward, jump)
    this.nRecurrentState = 4; // extra recurrent states for feedback.
    this.nOutput = this.nGameOutput + this.nRecurrentState;
    this.nInput = this.nGameInput + this.nOutput;

    // store current inputs and outputs
    this.inputState = zeros(this.nInput);
    this.convInputState = new Vol(1, 1, this.nInput); // compatible with convnetjs lib input.
    this.outputState = zeros(this.nOutput);
    this.prevOutputState = zeros(this.nOutput);

    // setup neural network:
    this.layer_defs = [];
    this.layer_defs.push({
      type: "input",
      out_sx: 1,
      out_sy: 1,
      out_depth: this.nInput,
    });
    this.layer_defs.push({
      type: "fc",
      num_neurons: this.nOutput,
      activation: "tanh",
    });

    this.net = new Net();
    this.net.makeLayers(this.layer_defs);

    var chromosome = new convnetjs.Chromosome(initGene);

    chromosome.pushToNetwork(this.net);

    //convnetjs.randomizeNetwork(this.net); // set init settings to be random.
  }

  populate(chromosome: any) {
    // populate network with a given chromosome.
    chromosome.pushToNetwork(this.net);
  }

  arrayToString(x: number[], precision: number) {
    "use strict";
    var result = "[";
    for (var i = 0; i < x.length; i++) {
      result += Math.round(precision * x[i]) / precision;
      if (i < x.length - 1) {
        result += ",";
      }
    }
    result += "]";
    return result;
  }

  getInputStateString() {
    "use strict";
    return this.arrayToString(this.inputState, 100);
  }

  getOutputStateString() {
    "use strict";
    return this.arrayToString(this.outputState, 100);
  }

  setCurrentInputState(agent: any, opponent: any) {
    "use strict";
    var i;
    var scaleFactor = 10; // scale inputs to be in the order of magnitude of 10.
    var scaleFeedback = 1; // to scale back up the feedback.
    this.inputState[0] = agent.state.x / scaleFactor;
    this.inputState[1] = agent.state.y / scaleFactor;
    this.inputState[2] = agent.state.vx / scaleFactor;
    this.inputState[3] = agent.state.vy / scaleFactor;
    this.inputState[4] = agent.state.bx / scaleFactor;
    this.inputState[5] = agent.state.by / scaleFactor;
    this.inputState[6] = agent.state.bvx / scaleFactor;
    this.inputState[7] = agent.state.bvy / scaleFactor;
    this.inputState[8] = (0 * opponent.state.x) / scaleFactor;
    this.inputState[9] = (0 * opponent.state.y) / scaleFactor;
    this.inputState[10] = (0 * opponent.state.vx) / scaleFactor;
    this.inputState[11] = (0 * opponent.state.vy) / scaleFactor;
    for (i = 0; i < this.nOutput; i++) {
      // feeds back output to input
      this.inputState[i + this.nGameInput] =
        this.outputState[i] * scaleFeedback * 1;
    }

    for (i = 0; i < this.nInput; i++) {
      // copies input state into convnet cube object format to be used later.
      this.convInputState.w[i] = this.inputState[i];
    }
  }

  forward() {
    "use strict";
    // get output from neural network:
    var a = this.net.forward(this.convInputState);
    for (var i = 0; i < this.nOutput; i++) {
      this.prevOutputState[i] = this.outputState[i]; // backs up previous value.
      this.outputState[i] = a.w[i];
    }
  }
}
