const firebaseConfig = {
  apiKey: "AIzaSyD2rtOS7GcYURKouuO1DmPkGYaN3BiERL8",
  authDomain: "brain-tumor-2ea99.firebaseapp.com",
  databaseURL: "https://brain-tumor-2ea99-default-rtdb.firebaseio.com",
  projectId: "brain-tumor-2ea99",
  storageBucket: "brain-tumor-2ea99.appspot.com",
  messagingSenderId: "393430095565",
  appId: "1:393430095565:web:67482c41fec8524a7bbb84",
  measurementId: "G-1EXK7R1VSG"
};

if (!firebase.apps.length) {
  firebase.initializeApp(firebaseConfig);
}

const storageRef = firebase.storage().ref();
const firestore = firebase.firestore();
const auth = firebase.auth();
let userId = '';

auth.onAuthStateChanged(function(user) {
  if (user) {
    userId = user.uid;
    console.log("User ID: ", userId);
    initializeApp();
  } else {
    console.log("No user logged in");
    redirectToLogin();
  }
});

function checkUserState() {
  const user = auth.currentUser;
  if (user) {
    userId = user.uid;
    console.log("User ID: ", userId);
    initializeApp();
  } else {
    console.log("No user logged in");
    redirectToLogin();
  }
}

function initializeApp() {
  // Firestore 모드 확인
  firestore.collection('test').get().catch((error) => {
    if (error.code === 'failed-precondition') {
      console.error("Firestore is in Datastore mode, which is not supported.");
      alert("Firestore가 Datastore 모드로 설정되어 있습니다. Native 모드로 변경해야 합니다.");
      return;  // 더 이상의 초기화 작업을 진행하지 않도록 함
    } else {
      console.error("다른 Firestore 에러:", error);
    }
  });

  // List of functions that depend on userId being set
  listDocuments();
}


function redirectToLogin() {
  // Redirect to login page or display a message
  alert("You must be logged in to use this application.");
  window.location.href = '/login'; // Redirect to your login page
}

function logoutUser() {
  auth.signOut().then(() => {
    window.location.href = '/main';
  }).catch((error) => {
    console.error('Logout failed:', error);
  });
}

function listDocuments() {
  if (!userId) {
    console.error('User ID가 설정되지 않았습니다.');
    return;
  }

  const collectionName = `${userId}`;
  const docListElement = document.getElementById('docList');
  docListElement.innerHTML = '';

  firestore.collection(collectionName).get()
    .then((querySnapshot) => {
      querySnapshot.forEach((doc) => {
        const li = document.createElement('li');
        li.textContent = doc.id;
        li.onclick = () => {
          document.getElementById('docIdSearch').value = doc.id;
          searchDocument();
        };
        docListElement.appendChild(li);
      });
      docListElement.style.display = 'block';
    })
    .catch((error) => {
      console.error('Error listing documents:', error);
    });
}

document.querySelectorAll('.custom-file-input').forEach(input => {
  input.addEventListener('change', function(e) {
    const fileName = e.target.files[0].name;
    const nextSibling = e.target.nextElementSibling;
    nextSibling.innerText = fileName;
  });
});

function searchDocument() {
  if (!userId) {
    console.error('User ID가 설정되지 않았습니다.');
    return;
  }

  const docId = document.getElementById('docIdSearch').value;

  if (!docId) {
    alert('Please enter the Document ID.');
    return;
  }

  clearResults();

  const collectionName = `${userId}`;
  firestore.collection(collectionName).doc(docId).get()
    .then((doc) => {
      if (doc.exists) {
        const data = doc.data();

        document.getElementById('uploadedImageDisplay').style.display = 'none';
        document.getElementById('resultsDisplay').style.display = 'flex';
        document.getElementById('originalImageContainer').innerHTML = `<img src="${data.imageUrl}" alt="Uploaded Image" style="width: 100%; height: auto;">`;

        document.getElementById('resultContainer').innerHTML = `
          <div class="result-card">
            <h3>VGG16 Model Result</h3>
            <p>Result: ${data.vgg16Result || 'N/A'}</p>
            <p>Confidence: ${data.vgg16Confidence || 'N/A'}</p>
          </div>
          <div class="result-card">
            <h3>ResNet101 Model Result</h3>
            <p>Result: ${data.resnet101Result || 'N/A'}</p>
            <p>Confidence: ${data.resnet101Confidence || 'N/A'}</p>
          </div>
          <div class="result-card">
            <h3>AlexNet Model Result</h3>
            <p>Result: ${data.alexnetResult || 'N/A'}</p>
            <p>Confidence: ${data.alexnetConfidence || 'N/A'}</p>
          </div>
        `;
      } else {
        alert('No document found with that ID.');
      }
    })
    .catch((error) => {
      console.error('Error getting document:', error);
    });
}

async function analyzePneumoniaAndDICOM() {
  if (!userId) {
    console.error('User ID가 설정되지 않았습니다.');
    alert('You must be logged in to analyze the image.');
    return;
  }

  clearResults();

  const fileInput = document.getElementById('pneumoniaImageInput');
  const file = fileInput.files[0];
  if (!file) {
    alert('파일을 업로드해주세요.');
    console.error('파일을 업로드하지 않았습니다.');
    return;
  }

  const patientName = document.getElementById('patientName').value;
  const patientDOBElement = document.getElementById('patientDOB');
  if (!patientName || !patientDOBElement.value) {
    alert('모든 정보를 입력해주세요.');
    console.error('모든 정보를 입력하지 않았습니다.');
    return;
  }

  const patientDOB = patientDOBElement.value;
  const [year, month, day] = patientDOB.split('-');
  const twoDigitYear = year.substring(2);
  const dobFormatted = `${twoDigitYear}${month}${day}`;
  const docId = `${patientName}${dobFormatted}`;

  const formData = new FormData();
  formData.append('file', file);

  const fileExtension = file.name.split('.').pop().toLowerCase();
  let vggEndpoint = 'http://192.168.0.13:8000/judgePN-fromimg-with-vgg16/';
  let resnetEndpoint = 'http://192.168.0.13:8000/judgePN-fromimg-with-resnet101/';
  let alexnetEndpoint = 'http://192.168.0.13:8000/judgePN-fromimg-with-alexnet-web/';

  if (fileExtension === 'dcm') {
    vggEndpoint = 'http://192.168.0.13:8000/judgePN-fromdcm-web/';
  }

  console.log(`VGG Endpoint: ${vggEndpoint}`);
  console.log(`ResNet Endpoint: ${resnetEndpoint}`);
  console.log(`AlexNet Endpoint: ${alexnetEndpoint}`);
  console.log(`File name: ${file.name}`);

  try {
    const vggData = await sendAnalysisRequest(vggEndpoint, formData);
    let jpegDownloadURL = null;
    let downloadURL = null; // Define downloadURL here

    if (fileExtension === 'dcm') {
      const jpegBlob = base64ToBlob(vggData.image, 'image/jpeg');
      const jpegFile = new File([jpegBlob], `${docId}.jpeg`, { type: 'image/jpeg' });
      const jpegFileRef = storageRef.child(`${userId}/${jpegFile.name}`);
      const jpegSnapshot = await jpegFileRef.put(jpegFile);
      jpegDownloadURL = await jpegSnapshot.ref.getDownloadURL();
      console.log("JPEG URL: ", jpegDownloadURL);
    }

    const resnetData = await sendAnalysisRequest(resnetEndpoint, formData);
    const alexnetData = await sendAnalysisRequest(alexnetEndpoint, formData);

    displayResults(jpegDownloadURL || URL.createObjectURL(file), vggData, resnetData, alexnetData);

    await uploadToFirebaseStorage(file, docId, vggData, resnetData, alexnetData, jpegDownloadURL || downloadURL);
  } catch (error) {
    console.error('오류 발생:', error);
  }
}

async function sendAnalysisRequest(endpoint, formData) {
  console.log(`Sending request to ${endpoint}`);
  const response = await fetch(endpoint, {
    method: 'POST',
    body: formData,
  });
  console.log(`응답 상태: ${response.status}`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const data = await response.json();
  console.log(`응답 데이터: `, data);
  return data;
}

async function uploadToFirebaseStorage(file, docId, vggData, resnetData, alexnetData, jpegURL) {
  try {
    console.log('파일을 Firebase Storage에 업로드합니다.');
    const fileRef = storageRef.child(`${userId}/${file.name}`);
    const snapshot = await fileRef.put(file);
    console.log('Firebase Storage에 업로드 완료:', snapshot);
    const downloadURL = await snapshot.ref.getDownloadURL();
    console.log('다운로드 URL 얻기 완료:', downloadURL);
    const collectionName = `${userId}`;
    const currentTime = new Date().toISOString();

    const vggResult = vggData.result || 'No result';
    const vggConfidence = vggData.confidence !== undefined ? vggData.confidence : 'N/A';
    const resnetResult = resnetData.result || 'No result';
    const resnetConfidence = resnetData.confidence !== undefined ? resnetData.confidence : 'N/A';
    const alexnetResult = alexnetData.result || 'No result';
    const alexnetConfidence = alexnetData.confidence !== undefined ? alexnetData.confidence : 'N/A';

    await firestore.collection(collectionName).doc(docId).set({
      uploadTime: currentTime,
      imageUrl: jpegURL || downloadURL,
      fileName: file.name,
      vgg16Result: vggResult,
      vgg16Confidence: vggConfidence,
      resnet101Result: resnetResult,
      resnet101Confidence: resnetConfidence,
      alexnetResult: alexnetResult,
      alexnetConfidence: alexnetConfidence
    });
    console.log('Firestore에 문서 작성 완료');
  } catch (error) {
    console.error('파일 업로드 오류:', error);
  }
}

function displayResults(imageURL, vggData, resnetData, alexnetData) {
  document.getElementById('resultsDisplay').style.display = 'flex';
  document.getElementById('originalImageContainer').innerHTML = `<img src="${imageURL}" alt="Uploaded Image" style="width: 100%; height: auto;">`;

  document.getElementById('resultContainer').innerHTML = `
    <div class="result-card">
      <h3>VGG16 Model Result</h3>
      <p>Result: ${vggData.result || 'No result'}</p>
      <p>Confidence: ${vggData.confidence}</p>
    </div>
    <div class="result-card">
      <h3>ResNet101 Model Result</h3>
      <p>Result: ${resnetData.result || 'No result'}</p>
      <p>Confidence: ${resnetData.confidence}</p>
    </div>
    <div class="result-card">
      <h3>AlexNet Model Result</h3>
      <p>Result: ${alexnetData.result || 'No result'}</p>
      <p>Confidence: ${alexnetData.confidence}</p>
    </div>
  `;
}

function base64ToBlob(base64, mime) {
  const byteCharacters = atob(base64);
  const byteArrays = [];
  for (let offset = 0; offset < byteCharacters.length; offset += 512) {
    const slice = byteCharacters.slice(offset, offset + 512);
    const byteNumbers = new Array(slice.length);
    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }
  return new Blob(byteArrays, { type: mime });
}

function filterDocumentList() {
  const searchValue = document.getElementById('docIdSearch').value.toLowerCase();
  const docListItems = document.getElementById('docList').getElementsByTagName('li');

  for (let item of docListItems) {
    const itemText = item.textContent.toLowerCase();
    if (itemText.includes(searchValue)) {
      item.style.display = '';
    } else {
      item.style.display = 'none';
    }
  }
}

function performSegmentation() {
  clearResults();

  const fileInput = document.getElementById('segmentationImageInput');
  const file = fileInput.files[0];
  if (!file) {
    alert('파일을 업로드해주세요.');
    return;
  }

  const formData = new FormData();
  formData.append('file', file);
  const fileExtension = file.name.split('.').pop().toLowerCase();
  let url = 'http://192.168.0.13:8000/lung-image-mask/';
  if (fileExtension === 'dcm') {
    url = 'http://192.168.0.13:8000/segment-dicom/';
  }

  fetch(url, {
    method: 'POST',
    body: formData
  }).then(response => response.blob())
    .then(imageBlob => {
      const imageUrl = URL.createObjectURL(imageBlob);
      document.getElementById('resultsDisplay').style.display = 'flex';
      document.getElementById('imageContainer').innerHTML = `<img src="${imageUrl}" alt="Segmented Image" style="width: 50%; height: auto; margin-top: 20px;">`;
      document.getElementById('resultContainer').innerHTML = '';
    }).catch(error => {
      console.error('Error:', error);
      document.getElementById('resultsDisplay').style.display = 'block';
      document.getElementById('resultsDisplay').innerHTML = `<p>Error: ${error.message}</p>`;
    });
}

function clearResults() {
  document.getElementById('resultsDisplay').style.display = 'none';
  document.getElementById('originalImageContainer').innerHTML = '';
  document.getElementById('imageContainer').innerHTML = '';
  document.getElementById('resultContainer').innerHTML = '';
}
