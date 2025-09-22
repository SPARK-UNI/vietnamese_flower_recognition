const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const preview = document.getElementById("preview");
const result = document.getElementById("result");

const cameraBtn = document.getElementById("cameraBtn");
const captureBtn = document.getElementById("captureBtn");
const camera = document.getElementById("camera");
const canvas = document.getElementById("canvas");
let stream;

// Hiển thị ảnh preview
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = e => {
      preview.innerHTML = `<img src="${e.target.result}">`;
    };
    reader.readAsDataURL(file);
  }
});

// Gửi ảnh đi predict
uploadBtn.addEventListener("click", () => {
  const file = fileInput.files[0];
  if (!file) {
    alert("Hãy chọn ảnh!");
    return;
  }
  const formData = new FormData();
  formData.append("file", file);

  fetch("/predict", { method: "POST", body: formData })
    .then(res => res.json())
    .then(data => {
      result.innerText = `🌼 Loài hoa: ${data.class} (${data.confidence}%)`;
    });
});

// Mở camera
cameraBtn.addEventListener("click", async () => {
  stream = await navigator.mediaDevices.getUserMedia({ video: true });
  camera.srcObject = stream;
  captureBtn.style.display = "inline-block";
});

// Chụp ảnh từ camera
captureBtn.addEventListener("click", () => {
  const ctx = canvas.getContext("2d");
  canvas.width = camera.videoWidth;
  canvas.height = camera.videoHeight;
  ctx.drawImage(camera, 0, 0);
  canvas.toBlob(blob => {
    preview.innerHTML = `<img src="${URL.createObjectURL(blob)}">`;

    const formData = new FormData();
    formData.append("file", blob, "capture.jpg");

    fetch("/predict", { method: "POST", body: formData })
      .then(res => res.json())
      .then(data => {
        result.innerText = `🌼 Loài hoa: ${data.class} (${data.confidence}%)`;
      });
  }, "image/jpeg");
});
