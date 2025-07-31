console.log('✅ main.js loaded (updated UI behavior)');

let currentStream = null;

function stopWebcam() {
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
  }
  const uploadContent = document.getElementById('uploadContent');
  const video = uploadContent.querySelector('video');
  if (video) video.remove();
}

document.getElementById('fileButton').addEventListener('click', () => {
  document.getElementById('fileInput').click();
});
document.getElementById('fileInput').addEventListener('change', event => {
  stopWebcam();
  const file = event.target.files[0];
  const uploadContent = document.getElementById('uploadContent');
  uploadContent.innerHTML = '';
  if (!file) return;

  const url = URL.createObjectURL(file);
  if (file.type.startsWith('image/')) {
    const img = document.createElement('img');
    img.src = url;
    img.alt = file.name;
    uploadContent.appendChild(img);
  } else if (file.type.startsWith('video/')) {
    const video = document.createElement('video');
    video.src = url;
    video.controls = true;
    video.autoplay = false;
    uploadContent.appendChild(video);
  } else {
    uploadContent.textContent = 'Unsupported file type.';
  }
  const wc = document.getElementById('webcamCaptureButton');
  wc.textContent = 'Webcam / Capture';
});

const webcamBtn = document.getElementById('webcamCaptureButton');
webcamBtn.addEventListener('click', () => {
  const uploadContent = document.getElementById('uploadContent');

  if (!currentStream) {
    // start webcam
    uploadContent.innerHTML = '';
    const video = document.createElement('video');
    video.autoplay = true;
    video.setAttribute('playsinline', '');
    video.style.maxHeight = '100%';
    uploadContent.appendChild(video);

    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        currentStream = stream;
        video.srcObject = stream;
        webcamBtn.textContent = 'Capture';
      })
      .catch(err => {
        console.error('Webcam error:', err);
        uploadContent.innerHTML = `<p style="color:salmon;">⚠️ Camera error: ${err.name}</p>`;
      });
  } else {
    const video = uploadContent.querySelector('video');
    if (!video) {
      stopWebcam();
      webcamBtn.textContent = 'Webcam / Capture';
      return;
    }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const img = document.createElement('img');
    img.src = canvas.toDataURL('image/png');
    img.alt = 'Captured still';
    img.style.maxWidth = '100%';
    img.style.borderRadius = '6px';

    stopWebcam();
    uploadContent.innerHTML = '';
    uploadContent.appendChild(img);
    webcamBtn.textContent = 'Webcam / Capture';
  }
});

// Predict stub
document.getElementById('predictButton').addEventListener('click', () => {
  const out = document.getElementById('outputContent');
  out.innerHTML = '<p style="opacity:.85;">[Predict clicked — model logic placeholder]</p>';
});

document.getElementById('clearButton').addEventListener('click', () => {
  document.getElementById('outputContent').innerHTML = '';
});