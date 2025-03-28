document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("file-input");
    const previewImage = document.getElementById("preview-image");

    // Show image preview when user selects a file
    fileInput.addEventListener("change", function () {
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                previewImage.src = e.target.result;
                previewImage.style.display = "block";
            };
            reader.readAsDataURL(file);
        }
    });
});
