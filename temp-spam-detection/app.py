# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import skin_cancer_detection as SCD

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def runhome():

    return render_template("home.html")


# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.


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

    for pic in pics:
        inputimg = Image.open(pic)
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
        
        results.append({
            "result": result,
            "info": info,
            "confidence_scores": confidence_scores,
            "max_prob": round(float(max_prob)*100, 2)
        })

    if len(results) == 1:
        return render_template("reults.html", result=results[0]['result'], info=results[0]['info'], 
                             confidence_scores=results[0]['confidence_scores'], 
                             max_prob=results[0]['max_prob'])
    else:
        return render_template("multi_results.html", results=results)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.

# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.
