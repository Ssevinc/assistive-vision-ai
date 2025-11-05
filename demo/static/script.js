const video = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");

async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
  video: { facingMode: { exact: "environment" } } });

    video.srcObject = stream;
    video.play();
    statusText.textContent = "Camera running...";
    detectLoop();
  } catch (err) {
    statusText.textContent = "Error accessing camera.";
    console.error(err);
  }
}

async function detectLoop() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageData = canvas.toDataURL("image/jpeg");

  // send frame to backend
  const res = await fetch("/detect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageData }),
  });

  const data = await res.json();
  console.log("Detections:", data.detections);
  statusText.textContent = `Detections: ${JSON.stringify(data.detections)}`;

  // repeat
  requestAnimationFrame(detectLoop);
}

setupCamera();
