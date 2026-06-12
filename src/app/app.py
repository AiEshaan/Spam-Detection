# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.
from flask import Flask, request, render_template, send_file, make_response
from PIL import Image
import numpy as np
import sys
import os

# Add project root to sys.path to find src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model import skin_cancer_detection as SCD

import tempfile
from io import StringIO, BytesIO
import csv
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet

# Set template folder path explicitly
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))


@app.route("/", methods=["GET", "POST"])
def runhome():
    return render_template("home.html")


def get_result_info(class_ind):
    if class_ind == 0:
        return """Actinic keratosis also known as solar keratosis or senile keratosis are names given to intraepithelial keratinocyte dysplasia. As such they are a pre-malignant lesion or in situ squamous cell carcinomas and thus a malignant lesion.
        <br><br><strong>Causes:</strong> Prolonged UV exposure
        <br><strong>Symptoms:</strong> Rough, scaly patches on sun-exposed skin
        <br><strong>Next Steps:</strong> Consult a dermatologist for possible biopsy or treatment"""
    elif class_ind == 1:
        return """Basal cell carcinoma is a type of skin cancer. Basal cell carcinoma begins in the basal cells — a type of cell within the skin that produces new skin cells as old ones die off.
        <br><br><strong>Causes:</strong> Prolonged sun exposure, fair skin, history of sunburns
        <br><strong>Symptoms:</strong> Pearly or waxy bumps, flat, flesh-colored or brown scar-like lesions
        <br><strong>Next Steps:</strong> See a dermatologist promptly for evaluation and treatment"""
    elif class_ind == 2:
        return """Benign lichenoid keratosis (BLK) usually presents as a solitary lesion that occurs predominantly on the trunk and upper extremities in middle-aged women.
        <br><br><strong>Causes:</strong> Possibly related to regressing solar lentigines
        <br><strong>Symptoms:</strong> Solitary, pink to reddish-brown papules or plaques
        <br><strong>Next Steps:</strong> Monitor for changes, see a dermatologist if concerned"""
    elif class_ind == 3:
        return """Dermatofibromas are small, noncancerous (benign) skin growths that can develop anywhere on the body but most often appear on the lower legs, upper arms or upper back.
        <br><br><strong>Causes:</strong> Often develop after minor skin injuries (like insect bites or cuts)
        <br><strong>Symptoms:</strong> Firm, roundish nodules; usually pink, red, or brown
        <br><strong>Next Steps:</strong> Typically harmless, but can be removed if bothersome"""
    elif class_ind == 4:
        return """A melanocytic nevus (also known as nevocytic nevus, nevus-cell nevus and commonly as a mole) is a type of melanocytic tumor that contains nevus cells.
        <br><br><strong>Causes:</strong> Genetic predisposition, sun exposure
        <br><strong>Symptoms:</strong> Usually small, dark spots or raised growths
        <br><strong>Next Steps:</strong> Monitor for changes using ABCDE rule, see a dermatologist if any changes occur"""
    elif class_ind == 5:
        return """Pyogenic granulomas are skin growths that are small, round, and usually bloody red in color. They tend to bleed because they contain a large number of blood vessels.
        <br><br><strong>Causes:</strong> Often occur after injury or minor trauma; also common in pregnancy
        <br><strong>Symptoms:</strong> Red, moist papules that bleed easily
        <br><strong>Next Steps:</strong> Can be removed by a dermatologist if they bleed or are bothersome"""
    elif class_ind == 6:
        return """Melanoma, the most serious type of skin cancer, develops in the cells (melanocytes) that produce melanin — the pigment that gives your skin its color.
        <br><br><strong>Causes:</strong> UV exposure, fair skin, family history, many moles
        <br><strong>Symptoms:</strong> New, unusual growths or changes in existing moles (ABCDE rule)
        <br><strong>Next Steps:</strong> Seek immediate medical attention from a dermatologist or oncologist"""

@app.route("/showresult", methods=["GET", "POST"])
def show():
    pics = request.files.getlist("pic")
    results = []
    temp_files = []

    for pic in pics:
        # Save uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(pic.filename)[1])
        temp_file.close()
        pic.save(temp_file.name)
        temp_files.append(temp_file.name)
        
        inputimg = Image.open(temp_file.name)
        inputimg = inputimg.resize((28, 28))
        img = np.array(inputimg).reshape(-1, 28, 28, 3)
        predictions = SCD.model.predict(img)

        predictions = predictions[0]
        max_prob = max(predictions)
        class_ind = np.argmax(predictions)
        result = SCD.classes[class_ind]

        # Prepare confidence scores for all classes
        class_names = list(SCD.classes.values())
        confidence_scores = [{"class_name": name, "score": float(score)} for name, score in zip(class_names, predictions)]
        confidence_scores.sort(key=lambda x: x["score"], reverse=True)  # Sort by confidence descending

        info = get_result_info(class_ind)
        
        # Generate Grad-CAM
        try:
            heatmap = SCD.generate_gradcam(img, class_ind)
            superimposed_img = SCD.overlay_gradcam_on_image(temp_file.name, heatmap)
            gradcam_b64 = SCD.gradcam_to_base64(superimposed_img)
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            gradcam_b64 = None
        
        results.append({
            "result": result,
            "info": info,
            "confidence_scores": confidence_scores,
            "max_prob": round(float(max_prob)*100, 2),
            "gradcam_b64": gradcam_b64,
            "class_ind": class_ind
        })

    # Store results in session for export
    global export_data
    export_data = results

    if len(results) == 1:
        return render_template("reults.html", result=results[0]['result'], info=results[0]['info'], 
                             confidence_scores=results[0]['confidence_scores'], 
                             max_prob=results[0]['max_prob'],
                             gradcam_b64=results[0]['gradcam_b64'])
    else:
        return render_template("multi_results.html", results=results)


@app.route("/export/csv")
def export_csv():
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(["Index", "Prediction", "Confidence (%)"])
    for i, result in enumerate(export_data):
        cw.writerow([i+1, result['result'], result['max_prob']])
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=skin_cancer_results.csv"
    output.headers["Content-type"] = "text/csv"
    return output


@app.route("/export/pdf")
def export_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Skin Cancer Detection Report", styles['Title']))
    story.append(Spacer(1, 12))

    for i, result in enumerate(export_data):
        story.append(Paragraph(f"Image {i+1} Prediction", styles['Heading2']))
        story.append(Paragraph(f"Result: {result['result']}", styles['Normal']))
        story.append(Paragraph(f"Confidence: {result['max_prob']}%", styles['Normal']))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name='skin_cancer_results.pdf',
        mimetype='application/pdf'
    )

if __name__ == "__main__":
    export_data = []
    app.run(host="0.0.0.0", port=5000, debug=True)


# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.

# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.
