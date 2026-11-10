"use strict";
const assert = require("node:assert/strict");
const test = require("node:test");
const {harness, fixture} = require("./state_consistency.test.js");

const identity = () => [[[1,0],[0,1]],[[1,0],[0,1]]];
const metadata = {lattice:"FCC",axis:"110",layers:1,max_angle:90,default_angle:fixture().angle,
  presets:[{name:"Σ9",angle:fixture().angle,label:"Σ9"}],reference_labels:[],layer_spacing:1,axial_period:1};
const vector = repeat => ({lines:[{text:`P1→P2 · axial ${repeat >= 0 ? "+" : ""}${repeat}`,grain:null},
  {text:`G1 current (polar): repeat-${repeat}`,grain:0}],displacement:[1,0,0],length:repeat+1,cross_grain:false});
const atoms = () => [{position:[0,0],grain:0,layer:0,half_indices:[0,0,0]},
  {position:[1,0],grain:0,layer:0,half_indices:[2,0,0]}];

test("preset normalization precedes delayed rendering and near search", async () => {
  const h=harness(), a=h.app;
  a.setMetadata(metadata); a.state.angle=22; a.setPattern(fixture(22));
  a.state.nearEnabled=true; a.state.nearMethod="strain";
  a.changeAngle(38.94); h.flush(45); const render=h.last("render");
  h.flush(150); const search=h.last("near_search");
  assert.equal(render.request.angle,fixture().angle);
  assert.equal(search.request.angle,render.request.angle);
  h.reply(render,fixture()); await Promise.resolve();
  h.reply(search,[{label:"same angle",f1:identity()[0],f2:identity()[1]}]); await Promise.resolve();
  assert.equal(a.state.nearCell.label,"same angle");
});

test("cell mode cannot erase an applied fit through its shortcut entry point", () => {
  const h=harness(), a=h.app; a.setPattern(fixture());
  a.state.manual=Array.from({length:4},(_,i)=>({position:[i,i],layer:0,source:"CSL",endpoints:[[i,i],[i,i]]}));
  a.state.manualFit={cell:{f1:identity()[0],f2:identity()[1]}};
  a.state.manualOriginal=a.state.manual.map(v=>({...v,source:"local"}));
  const before=JSON.stringify(a.sessionState()); a.setMode("cell");
  assert.equal(JSON.stringify(a.sessionState()),before);
  a.updateSummary(); assert.equal(h.node("apply-strain").disabled,false);
});

test("appearance changes preserve a pending physical search, actual inputs reject it", async () => {
  for (const beforeRequest of [true,false]) {
    const h=harness(), a=h.app; a.setPattern(fixture());
    a.state.nearEnabled=true; a.state.nearMethod="strain"; a.searchNear();
    if (!beforeRequest) h.flush(150);
    h.node("g1-color").value="#009900"; h.node("g1-color").handlers.input();
    a.state.displayRotation=31; a.state.referenceAxes=false; a.state.visibleLayers=[new Set(),new Set()];
    if (beforeRequest) h.flush(150);
    assert(h.last("near_search"),"unrelated appearance must not suppress a request");
    h.reply(h.last("near_search"),[{label:"still valid",f1:identity()[0],f2:identity()[1]}]); await Promise.resolve();
    assert.equal(a.state.nearCell.label,"still valid");
    a.searchNear(); h.flush(150); const old=h.last("near_search");
    a.state.angle=22; h.reply(old,[{label:"wrong angle"}]); await Promise.resolve();
    assert.equal(a.state.nearCell,null,"the actual computed angle must be protected");
  }
});

test("PNG waits for the latest vector and follows changes during the wait", async () => {
  const h=harness(), a=h.app; a.setPattern(fixture()); a.state.atoms=atoms();
  const first=a.updateVector(); h.reply(h.last("vector"),vector(0)); await first;
  h.node("axial-repeat").value="1"; h.node("axial-repeat").handlers.change(); const old=h.last("vector");
  const exporting=a.exportPNG(); await Promise.resolve(); await Promise.resolve();
  assert.equal(h.exports,0,"an old readout cannot be exported during a newer calculation");
  h.node("axial-repeat").value="2"; h.node("axial-repeat").handlers.change(); const fresh=h.last("vector");
  h.reply(old,vector(1)); await Promise.resolve(); assert.equal(h.exports,0);
  h.reply(fresh,vector(2)); await exporting;
  assert.equal(h.exports,1); assert(h.textDraws.includes("P1→P2 · axial +2"));
  assert(!h.textDraws.includes("P1→P2 · axial +1"));
});

test("a failed current vector cannot export an older readout", async () => {
  const h=harness(), a=h.app; a.setPattern(fixture()); a.state.atoms=atoms();
  const initial=a.updateVector(); h.reply(h.last("vector"),vector(0)); await initial;
  h.node("axial-repeat").value="1"; h.node("axial-repeat").handlers.change();
  const exporting=a.exportPNG(); await Promise.resolve(); h.reject(h.last("vector")); await exporting;
  assert.equal(h.exports,0);
});

test("vector readout lines and endpoint labels follow the grain colors", async () => {
  const h=harness(), a=h.app; a.setPattern(fixture());
  a.state.colors=["#112233","#445566"];
  a.state.atoms=[{position:[0,0],grain:0,layer:0,half_indices:[0,0,0]},
    {position:[1,0],grain:1,layer:1,half_indices:[1,0,1]}];
  const reading=a.updateVector();
  h.reply(h.last("vector"),{lines:[
    {text:"Cross-grain P1→P2 · G1-A → G2-B · axial +0",grain:null},
    {text:"G1 current (polar) · full: a₀/2[1 1 2]",grain:0},
    {text:"G1 current (polar) · in-plane: a₀[0 0 1]",grain:0},
    {text:"G2 current (polar) · full: a₀/2[1 -1 2]",grain:1},
    {text:"G2 current (polar) · in-plane: a₀[0 0 1]",grain:1},
    {text:"Current |Δr|/a₀ = 1.0607 · in-plane |Δxy|/a₀ = 1.0000 · Δz/a₀ = +0.3536",grain:null}],
    displacement:[1,0,0.3536],length:1.0607,in_plane_length:1,split:true,cross_grain:true,vectors:{}});
  await reading;
  assert.equal(h.node("vector-annotation").hidden,false);
  const colors=text=>h.textColors.filter(d=>d.text===text).map(d=>d.color);
  await a.exportPNG();
  assert.equal(h.exports,1);
  assert.deepEqual(colors("P1"),["#112233"],"P1 takes its grain color");
  assert.deepEqual(colors("P2"),["#445566"],"P2 takes its grain color");
  assert.deepEqual(colors("G1 current (polar) · full: a₀/2[1 1 2]"),["#112233"]);
  assert.deepEqual(colors("G1 current (polar) · in-plane: a₀[0 0 1]"),["#112233"]);
  assert.deepEqual(colors("G2 current (polar) · full: a₀/2[1 -1 2]"),["#445566"]);
  assert.deepEqual(colors("Cross-grain P1→P2 · G1-A → G2-B · axial +0"),["#293b59"]);
  assert.deepEqual(colors("Current |Δr|/a₀ = 1.0607 · in-plane |Δxy|/a₀ = 1.0000 · Δz/a₀ = +0.3536"),["#293b59"]);
  assert.deepEqual(colors("View Δxy/a₀ = (1.0000, 0.0000)"),["#293b59"],"cross-grain readouts add displayed components");
  // A later grain color and display rotation apply to the existing measurement.
  h.node("g1-color").value="#009900"; h.node("g1-color").handlers.input();
  a.state.displayRotation=90;
  h.textColors.length=0; h.textDraws.length=0;
  await a.exportPNG();
  assert.equal(h.exports,2);
  assert.deepEqual(colors("G1 current (polar) · full: a₀/2[1 1 2]"),["#009900"]);
  assert.deepEqual(colors("G2 current (polar) · full: a₀/2[1 -1 2]"),["#445566"]);
  assert(h.textDraws.includes("View Δxy/a₀ = (0.0000, 1.0000)"));
  assert(!h.textDraws.includes("View Δxy/a₀ = (1.0000, 0.0000)"));
});

test("legacy vector replies without readout lines still show the annotation", async () => {
  const h=harness(), a=h.app; a.setPattern(fixture()); a.state.atoms=atoms();
  const reading=a.updateVector();
  h.reply(h.last("vector"),{vectors:{G1:{current_formatted:"a₀[1 0 0]",strained:false}},length:1});
  await reading;
  assert.equal(h.node("vector-annotation").hidden,false);
  await a.exportPNG();
  assert.equal(h.exports,1);
});
