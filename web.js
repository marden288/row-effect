const camera = document.querySelector("#camera");
const portal = document.querySelector("#portal");
const message = document.querySelector("#message");
const names = ["Grid", "Duotone", "Halftone", "Chromatic", "Thermal", "Vintage", "Frosted", "Rose"];
const filters = [
  "contrast(1.15) saturate(1.3)",
  "grayscale(1) contrast(2) sepia(1) hue-rotate(180deg)",
  "grayscale(1) contrast(1.8)",
  "saturate(2) hue-rotate(25deg) contrast(1.2)",
  "grayscale(1) sepia(1) hue-rotate(300deg) saturate(4)",
  "sepia(.8) contrast(1.1) saturate(.8)",
  "blur(5px) brightness(1.3) saturate(.6)",
  "sepia(.4) hue-rotate(290deg) saturate(2) contrast(1.2)",
];
let index = 0;
function updateFilter() {
  portal.style.setProperty("--portal-filter", filters[index]);
  document.querySelector("#filter-name").textContent = names[index];
  document.querySelector("#filter-count").textContent = `${index + 1} / ${names.length}`;
}
function move(step) {
  index = (index + step + filters.length) % filters.length;
  updateFilter();
}
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    camera.srcObject = stream;
    portal.srcObject = stream;
    message.hidden = true;
    document.querySelector("#start").textContent = "Camera enabled";
  } catch (error) {
    message.textContent = "Camera access was blocked. Allow it in your browser settings.";
    console.error("Unable to start camera:", error);
  }
}
document.querySelector("#start").addEventListener("click", startCamera);
document.querySelector("#previous").addEventListener("click", () => move(-1));
document.querySelector("#next").addEventListener("click", () => move(1));
let touchStart = 0;
document.querySelector(".stage").addEventListener("touchstart", (event) => { touchStart = event.changedTouches[0].clientX; });
document.querySelector(".stage").addEventListener("touchend", (event) => {
  const distance = event.changedTouches[0].clientX - touchStart;
  if (Math.abs(distance) > 40) move(distance < 0 ? 1 : -1);
});
updateFilter();
