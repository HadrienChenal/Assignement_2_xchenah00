## 2nd Home Assignment - Object detection on your own dataset

In this assignment, you will go through the whole workflow of object detection: you will collect your own images,
annotate them, train a YOLO object detector and use it to detect objects in a video.

No data are provided. You will create the dataset yourself and write all the scripts, starting from
a small skeleton of the video rendering script (`render.py`).

---

You will have to:

1. **Choose the objects you want to detect.** You can choose any objects you like, but you have to use
   at least two classes. Choose objects you have at hand and can photograph in different places. Some examples:
   - computer mouse and keyboard,
   - fruit (e.g. bananas, apples and oranges),
   - cups and bottles,
   - keys and wallets,
   - coins, ...

2. **Take photos of the objects.** Take roughly 50 to 100 photos of your objects. This is only a rough estimate, you can take more if you want.
   The more images you take, the better your final detector will be. Make the dataset as varied as possible:
   - different numbers of objects in a photo (a single object, several objects of one class, objects of different classes together),
   - different locations and backgrounds,
   - different distances and viewing angles,
   - different lighting conditions (but not in the dark),
   - partially occluded objects.

3. **Annotate the objects in the photos with bounding boxes.** Use one of the following tools.
   - **Roboflow** (recommended; web application, https://roboflow.com):
     - Create a free account and an Object Detection project, upload your photos and annotate them in the browser.
     - Export the dataset in the YOLO format for Ultralytics (YOLOv8/YOLO11).
       CAREFUL: on the free plan, your project and photos are public. Do not upload photos you do not want to share.
     - You can use automatic annotation to pre-label the images and then correct them.
   - **Label Studio** (runs locally on your computer):
     - Install and start it with `pip install label-studio` and `label-studio start`, then open http://localhost:8080
       in your browser. Create a project, choose the "Object Detection with Bounding Boxes" template, set your class
       names and import your photos.
     - When you are done, export the annotations in the YOLO format (or in JSON and convert them with your own script).
       CAREFUL: Label Studio adds a prefix to the names of uploaded images, check that your label files match your images.

4. **Split the data into training and validation sets** and create the dataset YAML file which you pass
   to the training command. If you use Roboflow, it can do both for you when exporting the dataset.

5. **Train a YOLO detector.** Experiment with different model versions and sizes and with
   training parameters like learning rate, batch size, augmentations, image size... You have to train it yourself using the Ultralytics package (not in Roboflow or other online services).

6. **Record a short video with your objects.** The video should be 10 to 20 seconds long and all your classes have to appear in it. Complete the script `render.py`, which runs your trained detector on every frame of the video, draws the detections (bounding box, class name and confidence) and saves the result as a new video. Do not change its command line arguments, we will run it as:
   ```
   python render.py --model best.pt --video <your_video> --output <output_video>
   ```

---

SUBMIT:
1. **Your dataset as a link.** Do not upload the images into the information system. Either:
   - upload the images and the YOLO labels (split into training and validation sets) with the dataset YAML file
     to a cloud storage (e.g. Nextcloud or Google Drive) and share it with a link, or
   - give us the link to your public Roboflow project.

   Make sure the link works without logging in and keep the data available until the assignment is graded.
2. **A single zip uploaded into the information system** containing:
   - a text file `dataset.txt` with the link to your dataset,
   - your completed `render.py` and all your other scripts (annotation conversion if you needed one, data splitting, ...),
   - the trained model (weights/best.pt; if you trained several models, submit the best one)
     and all its training outputs (*.png, *.jpg, results.csv, args.yaml),
   - your input video.

There will be **NO** consultation like in the first assignment. We will grade the assignment based on the submitted files.
We will not grade the quality of your dataset or the accuracy of your model, as long as they make sense
(a reasonable number of correctly annotated photos and a model which actually detects your objects).

---

Recommendations:
- Check your labels visually (draw the boxes into the images) before training. Errors in the conversion
  (e.g. swapped coordinates, corner instead of center, pixels instead of relative values) are very common.
- Pass the 'project=/path' option to the train command, it will save the results to the specified folder.
- When you run the training with e.g. model=yolo11n.pt, the API downloads the pretrained YOLO model.
  Your trained model will be saved in the output directory as weights/(last|best).pt.
- We recommend the nano or small model variants (e.g. yolo11n.pt, yolo11s.pt). You can also try the medium
  variant to see if it helps, but do not submit a medium (or larger) model.

---

Where to train:
- **Your own computer with an NVIDIA GPU:** install PyTorch with CUDA support (see https://pytorch.org/get-started/locally/)
  and check that `torch.cuda.is_available()` returns True. Ultralytics then uses the GPU automatically (or pass `device=0`).
  On a Mac with Apple Silicon, pass `device=mps`.
- **Your own computer without a GPU:** with a small dataset and the nano model, training on the CPU works, it is just
  slower. Start with a smaller image size and fewer epochs to check that everything works, then run the full training.
- **Google Colab** (https://colab.research.google.com): free GPU in the browser. Switch to a GPU runtime
  (Runtime -> Change runtime type), install Ultralytics with `!pip install ultralytics` and upload your dataset as a zip.
  The free GPU time is limited and all files are deleted when the session ends, so save your training outputs
  (e.g. to your Google Drive) before closing it.
- If you run out of GPU memory, lower the batch size (`batch=`) or the image size (`imgsz=`).

---

Resources:
- First home assignment -- for video reading and writing
- Ultralytics documentation:
    - https://docs.ultralytics.com/modes/train/
    - https://docs.ultralytics.com/modes/predict/
    - https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format

---

If you run into any trouble, ask on the course Discord channel. You can ask about anything, including
whether you are doing something correctly or whether your results look right. Don't hesitate to ask,
there are no stupid questions.

You can also send any questions, comments or bug reports to Martin Kostelnik (ikostelnik@fit.vutbr.cz).
