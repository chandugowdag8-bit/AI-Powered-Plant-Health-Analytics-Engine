AI-POWERED PLANT HEALTH ANALYTICS ENGINE — LOCAL REAL-AI STARTER
================================================================

PROJECT STATUS
- Existing website UI is preserved.
- Scanner uploads the selected image to the local FastAPI /predict endpoint.
- Backend loads a trained TensorFlow model when model files are present.
- Training script uses MobileNetV2 transfer learning and PlantVillage class folders.
- No dataset or trained model weights are included in this ZIP.

IMPORTANT LIMITATIONS
This is an academic prototype, not a field-validated agricultural diagnostic tool.
PlantVillage images are largely captured under controlled conditions. Accuracy on
real outdoor farm photos may be lower. A softmax confidence threshold does NOT
reliably detect unknown crops, non-leaf images, or diseases outside the dataset.
Do not use predictions alone to make pesticide or crop-treatment decisions.

FOLDER LAYOUT
plant_health_real_ai/
  index.html
  README.txt
  backend/
    main.py
    train_model.py
    requirements.txt
    dataset/
      PlantVillage/
        Apple___Apple_scab/
        Apple___Black_rot/
        ...

STEP 1 — EXTRACT
Extract this ZIP. Open Command Prompt in the folder that contains index.html
and the backend folder. Do not run commands from the outer download folder if
there is another nested project folder.

STEP 2 — DATASET
Download an image-folder version of PlantVillage. Dataset reference:
https://www.tensorflow.org/datasets/catalog/plant_village
Research reference:
Mohanty, Hughes and Salathé (2016), Using Deep Learning for Image-Based Plant
Disease Detection, https://doi.org/10.3389/fpls.2016.01419

Extract the class folders directly into:
backend/dataset/PlantVillage/

Correct example:
backend/dataset/PlantVillage/Apple___Apple_scab/image1.jpg
backend/dataset/PlantVillage/Apple___Black_rot/image2.jpg

Do not leave an extra PlantVillage/PlantVillage nesting level. The class folders
must be immediate children of backend/dataset/PlantVillage. Check the dataset
license and attribution terms for your use case.

STEP 3 — INSTALL PACKAGES (Python 3.13, 64-bit)
From the project root, run:
  py -m pip install -r backend\requirements.txt

The requirements are set for TensorFlow 2.21.0 and NumPy 2.x. If packages are
already installed, pip will report them as satisfied.

STEP 4 — TRAIN
From the project root, run:
  py backend\train_model.py

The script uses a 20% validation split, batch size 16 and avoids caching the full
image dataset in RAM. Training can take a long time on CPU. MobileNetV2 ImageNet
weights may need to download on the first run, so internet access is required.
It saves these files inside backend/:
  plant_disease_model.keras
  class_names.json

The reported validation accuracy is not a separate test-set result. For a strong
academic evaluation, create an independent test set and test on new field images.

STEP 5 — START API
From the project root:
  py -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

Open:
  http://127.0.0.1:8000/health
  http://127.0.0.1:8000/docs

After training and restarting the API, /health should report model_loaded: true.

STEP 6 — START WEBSITE
Open a SECOND Command Prompt in the project root:
  py -m http.server 5500

Open in Chrome:
  http://localhost:5500

Keep both the API and website server running. Go to the AI Scanner and upload a
JPG, PNG or WEBP leaf image under 5 MB.

DEMO LOGINS
Farmer: farmer@demo.com / farmer123
Admin: admin@planthealth.local / admin123
These are local demo credentials only, not secure production authentication.

TROUBLESHOOTING
- Missing dataset: check the exact backend/dataset/PlantVillage path.
- Model not found: run the training command successfully, then restart Uvicorn.
- Could not reach AI: make sure API is running on port 8000.
- Browser favicon 404: harmless; no favicon has been configured.
- TensorFlow oneDNN messages: informational startup messages, not necessarily errors.
- This local CORS configuration is for development only; restrict it before deployment.

NEXT IMPROVEMENTS
1. Independent test split and per-class precision, recall, F1 and confusion matrix.
2. Out-of-distribution checks and validation with real local farm photographs.
3. Expert-reviewed crop/disease treatment database.
4. Secure authentication, consent, model versioning and production hosting.
