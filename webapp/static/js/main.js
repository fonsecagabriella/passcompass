/* main.js ──────────────────────────────────────────────────────────
 *
 * 1. Fetch the feature schema from /features          (buildForm)
 * 2. Render:
 *      • categorical  → <select>
 *      • numeric      → <input type="range"> slider  (NEW)
 * 3. Live-update the number shown next to each slider
 * 4. Collect the form and POST /predict
 *
 * Author: ChatGPT facelift 2025-06-23
 *───────────────────────────────────────────────────────────────────*/

/* ── 1. Field labels (UI friendly) ─────────────────────────────── */
const fieldLabel = {
  school     : "School",
  course     : "Course",
  sex        : "Gender",
  age        : "Age (years)",
  address    : "Home address",
  famsize    : "Family size",
  Pstatus    : "Parents’ co-habitation",
  Medu       : "Mother’s education level",
  Fedu       : "Father’s education level",
  Mjob       : "Mother’s job",
  Fjob       : "Father’s job",
  reason     : "Reason for choosing school",
  guardian   : "Primary guardian",
  traveltime : "Daily travel time",
  studytime  : "Weekly study time",
  failures   : "Past class failures",
  schoolsup  : "Extra school support",
  famsup     : "Family study support",
  paid       : "Paid extra classes",
  activities : "Extracurricular activities",
  nursery    : "Attended nursery",
  higher     : "Wants higher education",
  internet   : "Internet at home",
  romantic   : "Romantic relationship",
  famrel     : "Family relationship quality",
  freetime   : "Free time after school",
  goout      : "Going-out frequency",
  Dalc       : "Week-day alcohol use",
  Walc       : "Weekend alcohol use",
  health     : "Current health status",
  absences   : "School absences",
};

/* ── 2. Value-labels for dropdowns ─────────────────────────────── */
const valueLabel = {
  school   : { GP: "Gabriel Pereira", MS: "Mousinho Silveira" },
  sex      : { F: "Female", M: "Male" },
  address  : { U: "Urban",  R: "Rural" },
  famsize  : { GT3: "> 3 people", LE3: "≤ 3 people" },
  Pstatus  : { T: "Together", A: "Apart" },
  yesNo    : { yes: "Yes", no: "No" },
};
const prettyChoice = (field, raw) =>
  (valueLabel[field]?.[raw]) ?? (valueLabel.yesNo?.[raw]) ?? raw;

/* ─── groups: assign each field to a section title ───────────── */
const groupMap = {
  /* Student profile */
  school: "Student profile", course: "Student profile", sex: "Student profile",
  age: "Student profile", address: "Student profile",

  /* Home & Support */
  famsize: "Home & Support", Pstatus: "Home & Support",
  guardian: "Home & Support", Medu: "Home & Support", Fedu: "Home & Support",
  schoolsup: "Home & Support", famsup: "Home & Support", paid: "Home & Support",
  internet: "Home & Support",

  /* Study habits */
  traveltime: "Study habits", studytime: "Study habits",
  failures: "Study habits", higher: "Study habits",

  /* Lifestyle */
  activities: "Lifestyle", nursery: "Lifestyle", romantic: "Lifestyle",
  famrel: "Lifestyle", freetime: "Lifestyle", goout: "Lifestyle",
  Dalc: "Lifestyle", Walc: "Lifestyle", health: "Lifestyle",
  absences: "Lifestyle", reason: "Lifestyle", Mjob: "Lifestyle", Fjob: "Lifestyle",
};

const formEl    = document.getElementById("form");
const outEl     = document.getElementById("out");
const btnEl     = document.getElementById("predictBtn");
const spinnerEl = btnEl.querySelector(".spinner");
const btnLabel  = btnEl.querySelector(".btn-label");

/* ───────────────────────────────────────────────────────────── */
async function buildForm() {
  const schema = await (await fetch("/features")).json();

  /* container map: groupName → DOM node (details element) */
  const containers = {};

  schema.forEach(col => {
    const groupName = groupMap[col.name] ?? "Other";
    if (!containers[groupName]) {
      /* create <details><summary>… */
      const details = document.createElement("details");
      if (groupName === "Student profile") details.open = true; // auto-open 1st
      details.innerHTML = `
        <summary>${groupName}</summary>
        <div class="grid-2col"></div>`;
      formEl.appendChild(details);
      containers[groupName] = details.querySelector(".grid-2col");
    }

    const id    = `id_${col.name}`;
    const label = fieldLabel[col.name] ?? col.name;

    /* numeric → range slider */
    if (col.kind === "numeric") {
      containers[groupName].insertAdjacentHTML("beforeend",
        buildRangeField(col, id, label));
      const slider = document.getElementById(id);
      const out    = document.getElementById(`${id}_val`);
      slider.addEventListener("input", e => out.textContent = e.target.value);
      return;
    }

    /* categorical → select */
    if (col.kind === "categorical") {
      containers[groupName].insertAdjacentHTML("beforeend",
        buildSelectField(col, id, label));
      return;
    }

    /* fallback */
    containers[groupName].insertAdjacentHTML("beforeend", `
      <div class="field">
        <label for="${id}">${label}</label>
        <input type="text" id="${id}" name="${col.name}" required>
      </div>`);
  });
}

/* ─── HTML builders ─────────────────────────────────────────── */
function buildRangeField(col, id, label){
  const min = col.min ?? 0;
  const max = col.max ?? 10;
  const mid = Math.round((min + max) / 2);

  return `
    <div class="field">
      <label for="${id}">${label} <small>(${min}-${max})</small></label>
      <input  type="range" id="${id}" name="${col.name}"
              min="${min}" max="${max}" value="${mid}" step="1" required>
      <span class="range-value" id="${id}_val">${mid}</span>
    </div>`;
}
function buildSelectField(col, id, label){
  const opts = col.choices.map(v =>
    `<option value="${v}">${prettyChoice(col.name, v)}</option>`).join("");
  return `
    <div class="field">
      <label for="${id}">${label}</label>
      <select id="${id}" name="${col.name}" required>
        <option value="" disabled selected>Choose…</option>
        ${opts}
      </select>
    </div>`;
}

/* ─── submit handler ────────────────────────────────────────── */
document.addEventListener("submit", async e => {
  e.preventDefault();

  const payload = Object.fromEntries(new FormData(formEl).entries());

  /* disable button + show spinner */
  btnEl.disabled = true;
  spinnerEl.classList.remove("hidden");
  btnLabel.textContent = "Predicting…";

  const res  = await fetch("/predict", {
    method : "POST",
    headers: { "Content-Type": "application/json" },
    body   : JSON.stringify(payload)
  });

  const json = await res.json();
  outEl.textContent = JSON.stringify(json, null, 2);
  outEl.classList.remove("hidden");

  btnEl.disabled = false;
  spinnerEl.classList.add("hidden");
  btnLabel.textContent = "Predict";
});

/* bootstrap */
buildForm();