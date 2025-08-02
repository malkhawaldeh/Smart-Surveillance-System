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

// File / video selection
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

// Combined webcam / capture toggle
const webcamBtn = document.getElementById('webcamCaptureButton');
webcamBtn.addEventListener('click', () => {
  const uploadContent = document.getElementById('uploadContent');

  if (!currentStream) {
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
    const uploadContent = document.getElementById('uploadContent');
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

// Helper to collect current input to send
async function collectInputPackage() {
  const fileInput = document.getElementById('fileInput');
  const uploadContent = document.getElementById('uploadContent');
  const video = uploadContent.querySelector('video');
  const img = uploadContent.querySelector('img');

  if (fileInput.files[0]) {
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    return { body: formData, isForm: true };
  } else if (img) {
    return { body: JSON.stringify({ snapshot: img.src }), isForm: false };
  } else if (video) {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/png');
    return { body: JSON.stringify({ snapshot: dataURL }), isForm: false };
  } else {
    return null;
  }
}

// Predict button logic
document.getElementById('predictButton').addEventListener('click', async () => {
  const output = document.getElementById('outputContent');
  output.innerHTML = '<p style="opacity:.8;">Running inference...</p>';

  const pkg = await collectInputPackage();
  if (!pkg) {
    output.innerHTML = '<p style="color:salmon;">No input to predict on.</p>';
    return;
  }

  try {
    let response;
    if (pkg.isForm) {
      response = await fetch('/predict', {
        method: 'POST',
        body: pkg.body
      });
    } else {
      response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: pkg.body
      });
    }

    if (!response.ok) {
      const err = await response.json();
      output.innerHTML = `<pre style="color:salmon;">Error: ${JSON.stringify(err, null, 2)}</pre>`;
      return;
    }

    const data = await response.json();
    output.innerHTML = formatResult(data.result);
  } catch (e) {
    console.error('Predict failed', e);
    output.innerHTML = `<p style="color:salmon;">Request failed.</p>`;
  }
});

// Clear output
document.getElementById('clearButton').addEventListener('click', () => {
  document.getElementById('outputContent').innerHTML = '';
});

function formatResult(result) {
  if (!result) return '<p>No result.</p>';
  const container = document.createElement('div');
  container.style.display = 'flex';
  container.style.flexDirection = 'column';
  container.style.gap = '1rem';

  // Thresholds summary
  if (result.thresholds) {
    const threshCard = document.createElement('div');
    threshCard.style.padding = '10px';
    threshCard.style.borderRadius = '6px';
    threshCard.style.background = 'rgba(255,255,255,0.02)';
    threshCard.style.fontSize = '0.85rem';
    threshCard.innerHTML = `<div><strong>Applied Thresholds:</strong></div>
      <div>Face detection: ${result.thresholds.face_detection_conf}</div>
      <div>Person detection: ${result.thresholds.person_detection_conf}</div>`;
    container.appendChild(threshCard);
  }

  // Faces
  if (result.faces && result.faces.length) {
    const faceHeader = document.createElement('div');
    faceHeader.innerHTML = '<strong>Detected Faces</strong>';
    container.appendChild(faceHeader);
    result.faces.forEach((f, idx) => {
      const card = document.createElement('div');
      card.className = 'result-card';
      card.style.padding = '12px';
      card.style.borderRadius = '8px';
      card.style.background = 'rgba(255,255,255,0.03)';
      card.style.display = 'flex';
      card.style.flexWrap = 'wrap';
      card.style.gap = '12px';

      const meta = document.createElement('div');
      meta.style.flex = '1 1 200px';
      meta.innerHTML = `<div><strong>Face #${idx + 1}</strong></div>`;
      card.appendChild(meta);

      if (f.attributes && f.attributes.length) {
        const attrsDiv = document.createElement('div');
        attrsDiv.style.flex = '1 1 250px';
        attrsDiv.innerHTML = `<div><strong>Attributes</strong></div>`;
        const list = document.createElement('ul');
        list.style.margin = '4px 0';
        list.style.paddingLeft = '16px';
        f.attributes.forEach(attr => {
          const li = document.createElement('li');
          li.textContent = attr;
          list.appendChild(li);
        });
        attrsDiv.appendChild(list);
        card.appendChild(attrsDiv);
      }

      container.appendChild(card);
    });
  }

  // People
  if (result.people && result.people.length) {
    const peopleHeader = document.createElement('div');
    peopleHeader.innerHTML = '<strong>Detected People</strong>';
    container.appendChild(peopleHeader);
    result.people.forEach((p, idx) => {
      const card = document.createElement('div');
      card.className = 'result-card';
      card.style.padding = '12px';
      card.style.borderRadius = '8px';
      card.style.background = 'rgba(255,255,255,0.03)';
      card.style.display = 'flex';
      card.style.flexWrap = 'wrap';
      card.style.gap = '12px';

      const meta = document.createElement('div');
      meta.style.flex = '1 1 150px';
      meta.innerHTML = `<div><strong>Person #${idx + 1}</strong></div>`;
      card.appendChild(meta);

      if (p.fashion_upar && p.fashion_upar.length) {
        const fuDiv = document.createElement('div');
        fuDiv.style.flex = '1 1 250px';
        fuDiv.innerHTML = `<div><strong>Attributes</strong></div>`;
        const list = document.createElement('ul');
        list.style.margin = '4px 0';
        list.style.paddingLeft = '16px';
        p.fashion_upar.forEach(label => {
          const li = document.createElement('li');
          li.textContent = label;
          list.appendChild(li);
        });
        fuDiv.appendChild(list);
        card.appendChild(fuDiv);
      }

      container.appendChild(card);
    });
  }

  if ((!result.faces || !result.faces.length) && (!result.people || !result.people.length)) {
    const none = document.createElement('div');
    none.textContent = 'No detections.';
    container.appendChild(none);
  }

  return container.outerHTML;
}