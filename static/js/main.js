// 1) File‐select button → opens file dialog
document.getElementById('fileButton').addEventListener('click', () => {
  document.getElementById('fileInput').click();
});

// when a file is chosen, display it:
document.getElementById('fileInput').addEventListener('change', event => {
  const file = event.target.files[0];
  const uploadContent = document.getElementById('uploadContent');
  uploadContent.innerHTML = '';  // clear previous

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
    uploadContent.appendChild(video);
  }
  // TODO: HERE you could call your AI model API with `file`
});

// 2) Webcam button → starts live camera
document.getElementById('webcamButton').addEventListener('click', () => {
  const uploadContent = document.getElementById('uploadContent');
  uploadContent.innerHTML = '';
  const video = document.createElement('video');
  video.autoplay = true;
  uploadContent.appendChild(video);

  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => { video.srcObject = stream; })
    .catch(err => console.error('Webcam error:', err));
  
  // TODO: HERE you could capture frames & send to your AI model
});

// 3) Export button (stub)
document.getElementById('exportButton').addEventListener('click', () => {
  // TODO: Add export functionality here (e.g. download CSV/JSON of output)
  console.log('Export button clicked — wire up your export logic here.');
});

// 4) Clear button → clears output panel
document.getElementById('clearButton').addEventListener('click', () => {
  document.getElementById('outputContent').innerHTML = '';
});
