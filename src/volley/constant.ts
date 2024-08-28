import { zeros } from "@/utils/convnet/array";

export const INIT_GENE_JSON =
  '{"fitness":1.3846153846153846,"nTrial":0,"gene":{"0":7.5555,"1":4.5121,"2":2.357,"3":0.139,"4":-8.3413,"5":-2.36,"6":-3.3343,"7":0.0262,"8":-7.4142,"9":-8.0999,"10":2.1553,"11":2.4759,"12":1.5587,"13":-0.7062,"14":0.2747,"15":0.1406,"16":0.8988,"17":0.4121,"18":-2.082,"19":1.4061,"20":-12.1837,"21":1.2683,"22":-0.3427,"23":-6.1471,"24":5.064,"25":1.2345,"26":0.3956,"27":-2.5808,"28":0.665,"29":-0.0652,"30":0.1629,"31":-2.3924,"32":-3.9673,"33":-6.1155,"34":5.97,"35":2.9588,"36":6.6727,"37":-2.2779,"38":2.0302,"39":13.094,"40":2.7659,"41":-1.3683,"42":2.5079,"43":-2.6932,"44":-2.0672,"45":-4.2688,"46":-4.9919,"47":-1.1571,"48":-2.0693,"49":2.9565,"50":9.6875,"51":-0.7638,"52":-1.5896,"53":2.4563,"54":-2.5956,"55":-9.8478,"56":-4.9463,"57":-3.4502,"58":-3.0604,"59":-1.158,"60":6.3533,"61":16.0047,"62":1.4911,"63":7.9886,"64":2.3879,"65":-4.5006,"66":-1.8171,"67":0.9859,"68":-2.414,"69":-1.5698,"70":2.5173,"71":-8.6187,"72":-0.3068,"73":-3.6185,"74":-5.202,"75":-0.05,"76":7.2617,"77":-3.1099,"78":0.9881,"79":-0.5022,"80":1.6499,"81":2.1346,"82":2.8479,"83":2.1166,"84":-6.177,"85":0.2584,"86":-3.7623,"87":-4.8107,"88":-9.1331,"89":-2.9681,"90":-7.1177,"91":-1.4894,"92":-1.1885,"93":-4.1906,"94":-5.821,"95":-4.3202,"96":-1.4603,"97":2.3514,"98":-4.8101,"99":3.6935,"100":1.388,"101":3.2504,"102":6.6364,"103":-3.7216,"104":1.6191,"105":6.4388,"106":0.4765,"107":-4.4931,"108":-1.1007,"109":-4.3594,"110":-2.9777,"111":-0.3744,"112":3.5822,"113":3.9402,"114":-9.2382,"115":-4.3392,"116":0.2103,"117":-1.3699,"118":9.2494,"119":10.8483,"120":0.2389,"121":2.6535,"122":-8.2731,"123":-3.5133,"124":-5.0808,"125":3.0846,"126":-0.4851,"127":0.3938,"128":0.2459,"129":-0.3466,"130":-0.1684,"131":-0.7868,"132":-0.6009,"133":2.5491,"134":-3.2234,"135":-3.3352,"136":4.7229,"137":-4.1547,"138":3.6065,"139":-0.1261}}';

export const INIT_GENE_RAW = JSON.parse(INIT_GENE_JSON);

export const INIT_GENE = zeros(Object.keys(INIT_GENE_RAW.gene).length); // Float64 faster.

export const PLAYER_SPEED_X = 10 * 1.75;

export const SHOW_ARROW_KEYS = true;
export const REF_W = 24 * 2;
export const REF_H = REF_W;
export const REF_U = 1.5; // ground height
export const REF_WALLWIDTH = 1.0; // wall width
export const REF_WALLHEIGHT = 3.5;
export const FACTOR = 1;
export const PLAYER_SPEED_Y = 10 * 1.35;
export const MAX_BALL_SPEED = 15 * 1.5;

export const TIME_STEP = 1 / 30;
export const THE_FRAME_RATE = 60 * 1;
export const NUDGE = 0.1;
export const FRICTION = 1.0; // 1 means no friction, less means friction
export const WIND_DRAG = 1.0;
export const INIT_DELAY_FRAMES = 30 * 2 * 1;
export const TRAINING_FRAMES = 30 * 20; // assume each match is 7 seconds. (vs 30fps)
export const THE_GRAVITY = -9.8 * 2 * 1.5;
export const TRAINING_MODE = false;
export const HUMAN1 = false; // if this is true, then player 1 is controlled by keyboard
export const HUMAN2 = false; // same as above

export const BASE_SCORE_FONT_SIZE = 64;