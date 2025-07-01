/* main.js ──────────────────────────────────────────────────────────
 *
 * 1. Fetch feature schema, build collapsible form
 * 2. Random × Clear × Predict handlers
 * 3. Validation: alert missing fields
 * 4. Summary panel with chosen values + probability bar
 *3
 *───────────────────────────────────────────────────────────────────*/

/* ── field & choice labels (unchanged, trimmed) ───────────────── */
const fieldLabel = {
  school:"School", course:"Course", sex:"Gender", age:"Age (years)",
  address:"Home address", famsize:"Family size", Pstatus:"Parents’ co-habitation",
  Medu:"Mother’s education level", Fedu:"Father’s education level",
  Mjob:"Mother’s job", Fjob:"Father’s job", reason:"Reason for choosing school",
  guardian:"Primary guardian", traveltime:"Daily travel time",
  studytime:"Weekly study time", failures:"Past class failures",
  schoolsup:"Extra school support", famsup:"Family study support",
  paid:"Paid extra classes", activities:"Extracurricular activities",
  nursery:"Attended nursery", higher:"Wants higher education",
  internet:"Internet at home", romantic:"Romantic relationship",
  famrel:"Family relationship quality", freetime:"Free time after school",
  goout:"Going-out frequency", Dalc:"Week-day alcohol use",
  Walc:"Weekend alcohol use", health:"Current health status",
  absences:"School absences",
};
const valueLabel = {
  school:{ GP:"Gabriel Pereira", MS:"Mousinho Silveira" },
  sex:{ F:"Female", M:"Male" },
  address:{ U:"Urban", R:"Rural" },
  famsize:{ GT3:"> 3 people", LE3:"≤ 3 people" },
  Pstatus:{ T:"Together", A:"Apart" },
  yesNo:{ yes:"Yes", no:"No" },
  course:{ math:"Mathematics", por:"Portuguese" },
};
const prettyChoice = (f,v)=>(valueLabel[f]?.[v])??(valueLabel.yesNo?.[v])??v;

/* ── group map (sections) ─────────────────────────────────────── */
const groupMap = {
  school:"Student profile", course:"Student profile", sex:"Student profile",
  age:"Student profile", address:"Student profile",

  famsize:"Home & Support", Pstatus:"Home & Support",
  guardian:"Home & Support", Medu:"Home & Support", Fedu:"Home & Support",
  schoolsup:"Home & Support", famsup:"Home & Support", paid:"Home & Support",
  internet:"Home & Support",

  traveltime:"Study habits", studytime:"Study habits",
  failures:"Study habits", higher:"Study habits",

  activities:"Lifestyle", nursery:"Lifestyle", romantic:"Lifestyle",
  famrel:"Lifestyle", freetime:"Lifestyle", goout:"Lifestyle",
  Dalc:"Lifestyle", Walc:"Lifestyle", health:"Lifestyle",
  absences:"Lifestyle", reason:"Lifestyle", Mjob:"Lifestyle", Fjob:"Lifestyle",
};

/* ── DOM shortcuts ───────────────────────────────────────────── */
const formEl       = document.getElementById("form");
const btnEl        = document.getElementById("predictBtn");
const spinnerEl    = btnEl.querySelector(".spinner");
const btnLabel     = btnEl.querySelector(".btn-label");
const randomBtn    = document.getElementById("randomBtn");
const clearBtn     = document.getElementById("clearBtn");
const summaryWrap  = document.getElementById("summaryWrap");
const summaryData  = document.getElementById("summary-data");
const summaryRes   = document.getElementById("summary-result");

let SCHEMA=[];      // available globally

/* ────────────────────────────────────────────────────────────────
 * buildForm(): fetch schema → generate collapsible form
 *──────────────────────────────────────────────────────────────── */
async function buildForm(){
  SCHEMA = await (await fetch("/features")).json();
  const containers={};

  SCHEMA.forEach(col=>{
    /* 1 ▸ create <details> group if needed */
    const groupName = groupMap[col.name]??"Other";
    if(!containers[groupName]){
      const details=document.createElement("details");
      if(groupName==="Student profile") details.open=true;
      details.innerHTML=`
        <summary>${groupName}</summary>
        <div class="grid-2col"></div>`;
      formEl.appendChild(details);
      containers[groupName]=details.querySelector(".grid-2col");
    }

    const id=`id_${col.name}`;
    const label=fieldLabel[col.name]??col.name;

    /* 2 ▸ numeric → slider */
    if(col.kind==="numeric"){
      containers[groupName].insertAdjacentHTML("beforeend",
        buildRangeField(col,id,label));
      const slider=document.getElementById(id);
      const out   =document.getElementById(`${id}_val`);
      slider.addEventListener("input",e=>out.textContent=e.target.value);
      return;
    }
    /* 3 ▸ categorical → select */
    if(col.kind==="categorical"){
      containers[groupName].insertAdjacentHTML("beforeend",
        buildSelectField(col,id,label));
      return;
    }
    /* fallback */
    containers[groupName].insertAdjacentHTML("beforeend",`
      <div class="field">
        <label for="${id}">${label}</label>
        <input type="text" id="${id}" name="${col.name}" required>
      </div>`);
  });
}

/* ── HTML helpers ───────────────────────────────────────────── */
function buildRangeField(col,id,label){
  const min=col.min??0, max=col.max??10, mid=Math.round((min+max)/2);
  return `
    <div class="field">
      <label for="${id}">${label} <small>(${min}-${max})</small></label>
      <input type="range" id="${id}" name="${col.name}"
             min="${min}" max="${max}" value="${mid}" step="1" required>
      <span class="range-value" id="${id}_val">${mid}</span>
    </div>`;
}
function buildSelectField(col,id,label){
  const opts=col.choices.map(v=>
    `<option value="${v}">${prettyChoice(col.name,v)}</option>`).join("");
  return `
    <div class="field">
      <label for="${id}">${label}</label>
      <select id="${id}" name="${col.name}" required>
        <option value="" disabled selected>Choose…</option>${opts}
      </select>
    </div>`;
}

/* ── RANDOMISE form values ──────────────────────────────────── */
function randomizeForm(){
  SCHEMA.forEach(col=>{
    const el=document.getElementById(`id_${col.name}`); if(!el) return;
    if(col.kind==="categorical"){
      const rand=col.choices[Math.floor(Math.random()*col.choices.length)];
      el.value=rand; return;
    }
    if(col.kind==="numeric"){
      const min=col.min??0, max=col.max??10;
      const rand=Math.floor(Math.random()*(max-min+1))+min;
      el.value=rand;
      const span=document.getElementById(`id_${col.name}_val`);
      if(span) span.textContent=rand;
    }
  });
  summaryWrap.classList.add("hidden");  // hide previous result
}

/* ── CLEAR form values ──────────────────────────────────────── */
function clearForm(){
  SCHEMA.forEach(col=>{
    const el=document.getElementById(`id_${col.name}`); if(!el) return;
    if(col.kind==="numeric"){
      const mid=Math.round(((col.min??0)+(col.max??10))/2);
      el.value=mid;
      const span=document.getElementById(`id_${col.name}_val`);
      if(span) span.textContent=mid;
    }
    if(col.kind==="categorical"){ el.value=""; }
  });
  summaryWrap.classList.add("hidden");
}

/* ── SUMMARY builders ──────────────────────────────────────── */
function makeSummary(payload){
  const liHTML = SCHEMA.map(col=>{
    const val=payload[col.name];
    if(val===undefined||val==="") return "";
    const display=col.kind==="categorical"
      ? prettyChoice(col.name,val) : val;
    return `<li><strong>${fieldLabel[col.name]??col.name}:</strong> ${display}</li>`;
  }).join("");
  summaryData.innerHTML=`<ul>${liHTML}</ul>`;
}

function makeResult(prob){
  const p = Math.round(prob * 100);
  const isPass = p >= 50;

  /* assign pass / fail class to wrapper */
  summaryRes.className = `summary-result ${isPass ? "prob-pass" : "prob-fail"}`;

  summaryRes.innerHTML = `
    <p class="prob-label">${isPass ? "Likely to pass" : "Likely to fail"}</p>`;
    /*<div class="prob-bar" style="--p:${p}%"></div>
    <p style="margin-top:.4rem;font-size:.85rem;">Probability: ${p}%</p>`;*/
}

/* ── SUBMIT handler with validation + spinner + summary ────── */
document.addEventListener("submit",async e=>{
  e.preventDefault();

  /* validate categorical selects */
  const missing=[];
  SCHEMA.forEach(col=>{
    if(col.kind!=="categorical") return;
    const el=document.getElementById(`id_${col.name}`);
    if(el && el.value==="") missing.push(fieldLabel[col.name]??col.name);
  });
  if(missing.length){
    alert("Please fill in:\n\n"+missing.map(f=>"• "+f).join("\n")); return;
  }

  /* payload */
  const payload=Object.fromEntries(new FormData(formEl).entries());

  /* UI busy state */
  btnEl.disabled=true;
  spinnerEl.classList.remove("hidden");
  btnLabel.textContent="Predicting…";

  const res=await fetch("/predict",{
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body:JSON.stringify(payload)
  });
  const json=await res.json();

  /* update summary */
  makeSummary(payload);
  makeResult(json.probability ?? json.proba_pass ?? json.proba ?? 0);
  summaryWrap.classList.remove("hidden");

  /* reset button */
  btnEl.disabled=false;
  spinnerEl.classList.add("hidden");
  btnLabel.textContent="Predict";
});

/* ── bind random & clear ───────────────────────────────────── */
randomBtn.addEventListener("click",randomizeForm);
clearBtn .addEventListener("click",clearForm);

/* bootstrap */
buildForm();
