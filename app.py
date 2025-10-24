# app.py
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import os
import time
import numpy as np
import time
from apscheduler.schedulers.background import BackgroundScheduler

def cleanup_uploads(folder, max_age_hours=24):
    """
    folder: jisme files delete karni hain
    max_age_hours: kitne ghante se purani file delete ho
    """
    now = time.time()
    count_deleted = 0
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            file_age_hours = (now - os.path.getmtime(path)) / 3600
            if file_age_hours > max_age_hours:
                os.remove(path)
                count_deleted += 1
    if count_deleted > 0:
        print(f"[Cleanup] Deleted {count_deleted} old files from {folder}")

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'png', 'webp', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB limit (adjust if needed)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def make_output_filename(orig_name, suffix):
    base = secure_filename(os.path.splitext(orig_name)[0])
    ts = int(time.time() * 1000)
    return f"{base}_{suffix}_{ts}.png"  # use png for general compatibility

def processImage(filepath, operation):
    """
    filepath: full path to uploaded file
    operation: string selecting operation
    returns: relative path to processed file (e.g. static/processed/...)
    """
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not read image (cv2 returned None).")

    outname = None

    # operations
    if operation == "cgray":
        # convert to grayscale
        img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        outname = make_output_filename(os.path.basename(filepath), "gray")
        outpath = os.path.join(app.config['PROCESSED_FOLDER'], outname)
        cv2.imwrite(outpath, img_processed)
    elif operation == "cwebp":
        outname = make_output_filename(os.path.basename(filepath), "to_webp")
        outpath = os.path.join(app.config['PROCESSED_FOLDER'], outname)
        # encode as webp but save as png-named for browser compatibility; if you want .webp: change extension
        cv2.imwrite(outpath, img)
    elif operation == "cjpg":
        outname = make_output_filename(os.path.basename(filepath), "to_jpg")
        outpath = os.path.join(app.config['PROCESSED_FOLDER'], outname)
        cv2.imwrite(outpath, img)
    elif operation == "cpng":
        outname = make_output_filename(os.path.basename(filepath), "to_png")
        outpath = os.path.join(app.config['PROCESSED_FOLDER'], outname)
        cv2.imwrite(outpath, img)

    # additional operations
    elif operation == "rotate_90":
        img_processed = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        outname = make_output_filename(os.path.basename(filepath), "rot90")
        outpath = os.path.join(app.config['PROCESSED_FOLDER'], outname)
        cv2.imwrite(outpath, img_processed)
    elif operation == "rotate_180":
        img_processed = cv2.rotate(img, cv2.ROTATE_180)
        outname = make_output_filename(os.path.basename(filepath), "rot180")
        outpath = os.path.join(app.config['PROCESSED_FOLDER'], outname)
        cv2.imwrite(outpath, img_processed)
    elif operation == "rotate_270":
        img_processed = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        outname = make_output_filename(os.path.basename(filepath), "rot270")
        outpath = os.path.join(app.config['PROCESSED_FOLDER'], outname)
        cv2.imwrite(outpath, img_processed)
    elif operation == "flip_h":
        img_processed = cv2.flip(img, 1)
        outname = make_output_filename(os.path.basename(filepath), "fliph")
        outpath = os.path.join(app.config['PROCESSED_FOLDER'], outname)
        cv2.imwrite(outpath, img_processed)
    elif operation == "flip_v":
        img_processed = cv2.flip(img, 0)
        outname = make_output_filename(os.path.basename(filepath), "flipv")
        outpath = os.path.join(app.config['PROCESSED_FOLDER'], outname)
        cv2.imwrite(outpath, img_processed)
    elif operation.startswith("resize_"):
        # resize_x where x is percentage like 50,150
        try:
            pct = int(operation.split("_")[1])
            h, w = img.shape[:2]
            new_w = int(w * pct / 100)
            new_h = int(h * pct / 100)
            img_processed = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            outname = make_output_filename(os.path.basename(filepath), f"resize{pct}")
            outpath = os.path.join(app.config['PROCESSED_FOLDER'], outname)
            cv2.imwrite(outpath, img_processed)
        except Exception as e:
            raise
    elif operation == "blur":
        img_processed = cv2.GaussianBlur(img, (11,11), 0)
        outname = make_output_filename(os.path.basename(filepath), "blur")
        outpath = os.path.join(app.config['PROCESSED_FOLDER'], outname)
        cv2.imwrite(outpath, img_processed)
    elif operation == "bright_inc":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        v = cv2.add(v, 30)  # increase brightness
        final_hsv = cv2.merge((h,s,v))
        img_processed = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        outname = make_output_filename(os.path.basename(filepath), "bright_inc")
        outpath = os.path.join(app.config['PROCESSED_FOLDER'], outname)
        cv2.imwrite(outpath, img_processed)
    elif operation == "bright_dec":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        v = cv2.subtract(v, 30)  # decrease brightness
        final_hsv = cv2.merge((h,s,v))
        img_processed = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        outname = make_output_filename(os.path.basename(filepath), "bright_dec")
        outpath = os.path.join(app.config['PROCESSED_FOLDER'], outname)
        cv2.imwrite(outpath, img_processed)
    else:
        # unknown operation: just copy the file
        outname = make_output_filename(os.path.basename(filepath), "copy")
        outpath = os.path.join(app.config['PROCESSED_FOLDER'], outname)
        cv2.imwrite(outpath, img)

    # Return web-accessible path (relative)
    rel_path = f"{app.config['PROCESSED_FOLDER']}/{outname}"
    return rel_path

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/edit", methods=["GET", "POST"])
def edit():
    processed_relpath = None
    if request.method == "POST": 
        operation = request.form.get("operation")
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('home'))
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('home'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save using binary-safe method for non-ascii (Windows compatibility)
            file.save(upload_path)

            try:
                processed_relpath = processImage(upload_path, operation)
                flash("Image processed successfully.")
            except Exception as e:
                flash(f"Processing failed: {str(e)}")
                return render_template("index.html", processed_image=None)
        else:
            flash("Invalid file type. Allowed: " + ", ".join(ALLOWED_EXTENSIONS))
            return redirect(url_for('home'))

    return render_template("index.html", processed_image=processed_relpath)

if __name__ == '__main__':
    from apscheduler.schedulers.background import BackgroundScheduler
    import os

    # Schedule cleanup jobs
    scheduler = BackgroundScheduler()
    scheduler.add_job(lambda: cleanup_uploads("uploads", 24), 'interval', hours=6)
    scheduler.add_job(lambda: cleanup_uploads("static/processed", 24), 'interval', hours=6)
    scheduler.start()
    print("[Scheduler] Cleanup jobs scheduled every 6 hours.")

    # Run Flask app (production-friendly)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

