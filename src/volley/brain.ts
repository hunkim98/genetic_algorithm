import { zeros } from "@/utils/convnet/array";
import { Vol } from "@/utils/convnet/vol";
import { Net } from "@/utils/convnet/net";
import { Chromosome } from "@/utils/convnet/evolution/chromosome";
import { Agent } from "./agent";

// this is the first generation gene, which is a JSON string.
// we will create other chromosomes based on this init Gene JSON
// var initGeneJSON =
//   '{"fitness":1.3846153846153846,"nTrial":0,"gene":{"0":7.5555,"1":4.5121,"2":2.357,"3":0.139,"4":-8.3413,"5":-2.36,"6":-3.3343,"7":0.0262,"8":-7.4142,"9":-8.0999,"10":2.1553,"11":2.4759,"12":1.5587,"13":-0.7062,"14":0.2747,"15":0.1406,"16":0.8988,"17":0.4121,"18":-2.082,"19":1.4061,"20":-12.1837,"21":1.2683,"22":-0.3427,"23":-6.1471,"24":5.064,"25":1.2345,"26":0.3956,"27":-2.5808,"28":0.665,"29":-0.0652,"30":0.1629,"31":-2.3924,"32":-3.9673,"33":-6.1155,"34":5.97,"35":2.9588,"36":6.6727,"37":-2.2779,"38":2.0302,"39":13.094,"40":2.7659,"41":-1.3683,"42":2.5079,"43":-2.6932,"44":-2.0672,"45":-4.2688,"46":-4.9919,"47":-1.1571,"48":-2.0693,"49":2.9565,"50":9.6875,"51":-0.7638,"52":-1.5896,"53":2.4563,"54":-2.5956,"55":-9.8478,"56":-4.9463,"57":-3.4502,"58":-3.0604,"59":-1.158,"60":6.3533,"61":16.0047,"62":1.4911,"63":7.9886,"64":2.3879,"65":-4.5006,"66":-1.8171,"67":0.9859,"68":-2.414,"69":-1.5698,"70":2.5173,"71":-8.6187,"72":-0.3068,"73":-3.6185,"74":-5.202,"75":-0.05,"76":7.2617,"77":-3.1099,"78":0.9881,"79":-0.5022,"80":1.6499,"81":2.1346,"82":2.8479,"83":2.1166,"84":-6.177,"85":0.2584,"86":-3.7623,"87":-4.8107,"88":-9.1331,"89":-2.9681,"90":-7.1177,"91":-1.4894,"92":-1.1885,"93":-4.1906,"94":-5.821,"95":-4.3202,"96":-1.4603,"97":2.3514,"98":-4.8101,"99":3.6935,"100":1.388,"101":3.2504,"102":6.6364,"103":-3.7216,"104":1.6191,"105":6.4388,"106":0.4765,"107":-4.4931,"108":-1.1007,"109":-4.3594,"110":-2.9777,"111":-0.3744,"112":3.5822,"113":3.9402,"114":-9.2382,"115":-4.3392,"116":0.2103,"117":-1.3699,"118":9.2494,"119":10.8483,"120":0.2389,"121":2.6535,"122":-8.2731,"123":-3.5133,"124":-5.0808,"125":3.0846,"126":-0.4851,"127":0.3938,"128":0.2459,"129":-0.3466,"130":-0.1684,"131":-0.7868,"132":-0.6009,"133":2.5491,"134":-3.2234,"135":-3.3352,"136":4.7229,"137":-4.1547,"138":3.6065,"139":-0.1261}}';

// var initGeneRaw = JSON.parse(initGeneJSON);

// // initGene has the information of the first chromosome we will create
// var initGene = zeros(Object.keys(initGeneRaw.gene).length); // Float64 faster.
// for (var i = 0; i < initGene.length; i++) {
//   initGene[i] = initGeneRaw.gene[i];
// }

// design agent's brain using neural network
export class Brain {
  nGameInput: number;
  nGameOutput: number;
  nRecurrentState: number;
  nOutput: number;
  nInput: number;
  inputState: number[] | Float64Array;
  convInputState: any; // Replace with the appropriate type
  outputState: number[] | Float64Array;
  prevOutputState: number[] | Float64Array;
  layer_defs: Array<{
    type: string;
    [key: string]: any;
  }>; // Replace with the appropriate type
  net: Net; // Replace with the appropriate type

  constructor({ initGene }: { initGene: number[] | Float64Array }) {
    this.nGameInput = 12; // 8 states for agent, plus 4 state for opponent
    this.nGameOutput = 3; // 3 buttons (forward, backward, jump)
    this.nRecurrentState = 4; // extra recurrent states for feedback.
    this.nOutput = this.nGameOutput + this.nRecurrentState; // 3 buttons + 4 feedback states
    this.nInput = this.nGameInput + this.nOutput; // 12 states + 7 feedback states

    // store current inputs and outputs
    this.inputState = zeros(this.nInput);
    this.convInputState = new Vol(1, 1, this.nInput); // compatible with convnetjs lib input.
    this.outputState = zeros(this.nOutput);
    this.prevOutputState = zeros(this.nOutput);

    // setup neural network:
    this.layer_defs = [];
    this.layer_defs.push({
      type: "input",
      // out_sx is
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

    const chromosome = new Chromosome(initGene);

    chromosome.pushToNetwork(this.net);
  }

  // populate means to fill up the network with the chromosome
  // Remember that the one chromosome contains all the weights and biases of the network
  populate(chromosome: Chromosome) {
    // populate network with a given chromosome.
    chromosome.pushToNetwork(this.net);
  }

  // precision is the number of decimal places to round to.
  arrayToString(x: number[] | Float64Array, precision: number) {
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

  setCurrentInputState(agent: Agent, opponent: Agent) {
    "use strict";
    let i;
    const scaleFactor = 10; // scale inputs to be in the order of magnitude of 10.
    const scaleFeedback = 1; // to scale back up the feedback.
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
    const a = this.net.forward(this.convInputState);
    for (let i = 0; i < this.nOutput; i++) {
      this.prevOutputState[i] = this.outputState[i]; // backs up previous value.
      this.outputState[i] = a.w[i];
    }
  }
}
