/* main.js: builds the form dynamically and handles prediction */

const labelMap = {
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
  absences   : "School absences"
  // ↳ extend if your model expects more columns
};

async function buildForm() {
  const schema = await (await fetch("/features")).json();
  const form   = document.getElementById("form");

  schema.forEach(f => {
    let fieldHTML = "";

    const label = labelMap[f.name] ?? f.name;          // fall back to raw
    const id    = `id_${f.name}`;

    if (f.kind === "numeric") {
      fieldHTML = `
        <div class="field">
          <label for="${id}">${label}</label>
          <input type="number" id="${id}" name="${f.name}" step="any" required>
        </div>`;
    } else if (f.kind === "categorical") {
      const opts = f.choices
        .map(c => `<option value="${c}">${c}</option>`)
        .join("");
      fieldHTML = `
        <div class="field">
          <label for="${id}">${label}</label>
          <select id="${id}" name="${f.name}">
            ${opts}
          </select>
        </div>`;
    }
    form.insertAdjacentHTML("beforeend", fieldHTML);
  });

  form.insertAdjacentHTML("beforeend",
    `<button type="submit" class="btn">Predict</button>`);
}

buildForm();

// ───────────── submit handler ───────────────────────────────────────
document.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(e.target).entries());

  const res  = await fetch("/predict", {
    method : "POST",
    headers: { "Content-Type": "application/json" },
    body   : JSON.stringify(data)
  });

  document.getElementById("out").textContent =
    JSON.stringify(await res.json(), null, 2);
});
