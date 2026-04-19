function toggleDarkMode() {
    document.body.classList.toggle("dark");
}

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");

if (dropZone) {
    dropZone.onclick = () => fileInput.click();

    dropZone.ondrop = (e) => {
        e.preventDefault();
        fileInput.files = e.dataTransfer.files;
    };

    dropZone.ondragover = (e) => e.preventDefault();
}

const form = document.getElementById("uploadForm");

if (form) {
    form.addEventListener("submit", () => {

        document.getElementById("loader").style.display = "block";

        const bar = document.getElementById("progress-bar");
        const container = document.getElementById("progress-container");

        container.style.display = "block";

        let progress = 0;

        let interval = setInterval(() => {
            progress += 10;
            bar.style.width = progress + "%";

            if (progress >= 100) clearInterval(interval);
        }, 200);
    });
}