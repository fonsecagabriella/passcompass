/* main.js: builds the form dynamically and handles prediction */


/* Field labels */
const fieldLabel = {
  school     : "School",
  course     : "Course",
  sex        : "Gender",
  age        : "Age (years)",
  address    : "Home address",
  famsize    : "Family size",
  Pstatus    : "Parents’ cohabitation",
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
  // ↳ extend if your model expects more columns
};

/*  Choice-value labels  (add / extend freely) */
const valueLabel = {
  school   : { GP: "Gabriel Pereira", MS: "Mousinho Silveira" },
  sex      : { F: "Female", M: "Male" },
  address  : { U: "Urban",  R: "Rural" },
  famsize  : { GT3: " > 3 people", LE3: " ≤ 3 people" },
  Pstatus  : { T: "Together", A: "Apart" },
  Mjob     : { at_home: "At home", health: "Health", other: "Other",
                 services: "Services", teacher: "Teacher" },
  Fjob     : { at_home: "At home", health: "Health", other: "Other",
                 services: "Services", teacher: "Teacher" },
  Medu     : { 0: "None", 1: "Primary", 2: "Lower secondary",
                 3: "Upper secondary", 4: "Higher Education" },
  Fedu     : { "-": "None", 1: "Primary", 2: "Lower secondary",
                 3: "Upper secondary", 4: "Higher Education" },
  

  /* fall-back groups — reuse across many columns */
  yesNo    : { yes: "Yes", no: "No" },
  boolean  : { yes: "Yes", no: "No" },
};

/* Helper → picks the correct label map for a field */
function prettyChoice(field, raw) {
  const map =
    valueLabel[field]            // specific map
    ?? valueLabel.boolean        // common yes/no
    ?? {};
  return map[raw] ?? raw;        // fall back to raw value
}


async function buildForm() {
  const schema = await (await fetch("/features")).json();
  const form   = document.getElementById("form");

  schema.forEach(col => {
    const id    = `id_${col.name}`;
    const label = fieldLabel[col.name] ?? col.name;
    let html    = "";

    /* ─── numeric ───────────────────────────────────────── */
    if (col.kind === "numeric") {
      const min = col.min ?? "";
      const max = col.max ?? "";
      const rangeHint = (min || max) ? `(${min || "–"}–${max || "–"})` : "";

      html = `
        <div class="field">
          <label for="${id}">${label} <small>${rangeHint}</small></label>
          <input type="number"
                 id="${id}" name="${col.name}"
                 ${min && `min="${min}"`}
                 ${max && `max="${max}"`}
                 step="any" required>
        </div>`;
    }

    /* ─── categorical ───────────────────────────────────── */
    else if (col.kind === "categorical") {
      const opts = col.choices.map(v =>
        `<option value="${v}">${prettyChoice(col.name, v)}</option>`
      ).join("");

      html = `
        <div class="field">
          <label for="${id}">${label}</label>
          <select id="${id}" name="${col.name}" required>
            <option value="" disabled selected>Choose…</option>
            ${opts}
          </select>
        </div>`;
    }

    /* ─── unknown kinds – fallback to text ──────────────── */
    else {
      console.warn(`Unknown kind '${col.kind}' for '${col.name}'`);
      html = `
        <div class="field">
          <label for="${id}">${label}</label>
          <input type="text" id="${id}" name="${col.name}" required>
        </div>`;
    }

    form.insertAdjacentHTML("beforeend", html);
  });

  /*form.insertAdjacentHTML("beforeend",
    `<button type="submit" class="btn">Predict</button>`);*/
}

buildForm();

/* ─────────────────── submit handler ───────────────────── */
document.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(e.target).entries());

  const res = await fetch("/predict", {
    method : "POST",
    headers: { "Content-Type": "application/json" },
    body   : JSON.stringify(data)
  });

  document.getElementById("out").textContent =
    JSON.stringify(await res.json(), null, 2);
});