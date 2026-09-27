const fs = require("fs");
globalThis.window = globalThis;
globalThis.Graphics = { frameCount: 1, app: { ticker: {
  started: true, autoStart: true, maxFPS: 0, stop(){}, start(){}, update(){} } } };
globalThis.SceneManager = { _scene: { constructor: { name: "Scene_Map" } } };
globalThis.Input = { update(){} };
globalThis.$gameMap = { _mapId: 4 };
globalThis.$gamePlayer = { px: 0 };

eval(fs.readFileSync(process.argv[2], "utf8"));
const V = globalThis.__ash;

const shared = { tag: "SHARED" };
const root = {
  a: { "10": { child: shared }, "2": { child: shared } },
  b: { zzz: shared }
};

const enc = new V.Encoder();
const tree = enc.encode(root, 0);

const dec = new V.Decoder();
const back = dec.decode(tree);

const c10 = back.a["10"].child;
const c2 = back.a["2"].child;
const cz = back.b.zzz;

console.log(JSON.stringify({
  tags: [c10 && c10.tag, c2 && c2.tag, cz && cz.tag],
  allSame: c10 === c2 && c10 === cz,
  c10type: typeof c10,
  c2type: typeof c2
}));
