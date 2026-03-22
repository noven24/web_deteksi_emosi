// --- KONFIGURASI MODEL ---
// 👉 1. Sesuaikan ukuran ini dengan ukuran input saat training di Python
const IMAGE_SIZE = 256;

// 👉 2. Sesuaikan urutan kelas dengan urutan folder saat training
// (Biasanya TensorFlow mengurutkan berdasarkan abjad folder: Happy -> Sad)
const CLASSES = ['Senang', 'Sedih'];

// Variabel global
let model;
const modelPath = './tfjs_model/model.json';

// DOM Elements
const imageSelector = document.getElementById('image-selector');
const previewImage = document.getElementById('preview-image');
const placeholderText = document.getElementById('placeholder-text');
const statusText = document.getElementById('status-text');
const resultArea = document.getElementById('result-area');
const predictionText = document.getElementById('prediction-text');
const probabilityText = document.getElementById('probability-text');

// --- LANGKAH 1: MEMUAT MODEL ---
async function loadModel() {
    try {
        // Karena model dari Keras, kita menggunakan loadLayersModel
        model = await tf.loadLayersModel(modelPath);

        // Pemanasan model (Warming up) agar prediksi pertama lebih cepat
        const dummyInput = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
        model.predict(dummyInput).dispose();

        statusText.innerText = "Model AI Siap Digunakan!";
        statusText.className = "status-ready";
        console.log("Model loaded successfully");
    } catch (error) {
        statusText.innerText = "Gagal memuat model. Pastikan folder tfjs_model sudah benar.";
        statusText.style.color = "red";
        console.error(error);
    }
}

// --- LANGKAH 2: MENANGANI UNGGAHAN GAMBAR ---
imageSelector.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        // Tampilkan preview gambar
        previewImage.src = e.target.result;
        previewImage.classList.remove('hidden');
        placeholderText.classList.add('hidden');
        resultArea.classList.add('hidden'); // Sembunyikan hasil lama

        // Mulai prediksi setelah gambar dimuat sempurna di layar
        previewImage.onload = async () => {
            await predict();
        };
    };
    reader.readAsDataURL(file);
});

// --- LANGKAH 3: PRA-PEMROSESAN & PREDIKSI (VERSI ASYNC) ---
async function predict() {
    if (!model) return;

    statusText.innerText = "Sedang menganalisis ekspresi...";
    statusText.style.color = "#333";

    // 1. Siapkan tensor gambar di dalam tf.tidy agar memorinya aman
    const batchedImg = tf.tidy(() => {
        const imgTensor = tf.browser.fromPixels(previewImage);
        const resizedImg = tf.image.resizeNearestNeighbor(imgTensor, [IMAGE_SIZE, IMAGE_SIZE]);
        const normalizedImg = resizedImg.toFloat().div(tf.scalar(255));
        return normalizedImg.expandDims(0);
    });

    try {
        // 2. Eksekusi menggunakan predict (Wajib untuk Layers Model)
        const result = model.predict(batchedImg);

        // 3. Ambil angkanya
        const predictionsArray = await result.data();

        // 4. Bersihkan memori hasil eksekusi secara manual
        batchedImg.dispose();
        result.dispose();

        // 5. Tampilkan hasil
        displayResult(predictionsArray);

    } catch (error) {
        console.error("Error saat prediksi:", error);
        statusText.innerText = "Gagal memproses gambar. Lihat Console untuk detail.";
        statusText.style.color = "red";

        // Pastikan memori gambar tetap dibersihkan meski error
        batchedImg.dispose();
    }
}

// --- LANGKAH 4: MENAMPILKAN HASIL ---
function displayResult(predictions) {
    statusText.innerText = "Analisis selesai.";

    // Asumsi model outputnya Sigmoid (1 nilai antara 0-1) untuk binary classification
    // Jika nilai > 0.5 berarti kelas indeks ke-1 (Sedih), jika < 0.5 kelas indeks ke-0 (Senang)
    // 👉 SESUAIKAN LOGIKA INI JIKA OUTPUT MODELMU SOFTMAX (2 nilai)
    const probabilitasSedih = predictions[0];
    let classIndex;
    let confidence;

    if (probabilitasSedih > 0.5) {
        classIndex = 1; // Sedih
        confidence = probabilitasSedih;
        predictionText.style.color = "var(--sad-color)";
    } else {
        classIndex = 0; // Senang
        confidence = 1 - probabilitasSedih;
        predictionText.style.color = "var(--happy-color)";
    }

    // Tampilkan teks
    predictionText.innerText = CLASSES[classIndex];
    probabilityText.innerText = `Tingkat Keyakinan: ${(confidence * 100).toFixed(2)}%`;
    resultArea.classList.remove('hidden');
}

// Jalankan fungsi muat model saat halaman dibuka
loadModel();