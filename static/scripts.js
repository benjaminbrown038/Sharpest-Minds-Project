document.getElementById('upload-form').addEventListener('submit', async function(event) {
  event.preventDefault();

  const imageInput = document.getElementById('image-input');
  const image = imageInput.files[0];

  if (image) {
    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await fetch('/classify', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      document.getElementById('result').textContent = 'Prediction: ' + result.label;

      const reader = new FileReader();
      reader.onloadend = function() {
        document.getElementById('image-preview').style.display = 'block';
        document.getElementById('image-preview').src = reader.result;
      };
      reader.readAsDataURL(image);
    } catch (error) {
      console.error('Error during image classification:', error);
    }
  }
});
