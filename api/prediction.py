from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# Define class names
class_names = [
    "P00-P96 certain conditions originating in the perinatal period (ภาวะบางอย่างที่เกิดในระยะปริกำเนิด)",
    "A00-B99 certain infectious and parasitic diseases (โรคติดเชื้อและโรคปรสิตบางโรค)",
    "U00-U85 codes for special purposes (รหัสเพื่อวัตถุประสงค์พิเศษ)",
    "Q00-Q99 congenital malformations deformations and chromosomal abnormalities (รูปผิดปกติแต่กำเนิด รูปพิการ และความผิดปกติของโครโมโซม)",
    "D50-D89 diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism (โรคของเลือดและอวัยวะสร้างเลือดและความผิดปกติบางอย่างของกลไกภูมิคุ้มกัน)",
    "I00-I99 diseases of the circulatory system (โรคของระบบไหลเวียนโลหิต)",
    "K00-K95 diseases of the digestive system (โรคของระบบย่อยอาหาร)",
    "H60-H95 diseases of the ear and mastoid process (โรคของหูและปุ่มกระดูกกกหู)",
    "H00-H59 diseases of the eye and adnexa (โรคของตาและอวัยวะเคียงลูกตา)",
    "N00-N99 diseases of the genitourinary system (โรคของระบบสืบพันธุ์และระบบปัสสาวะ)",
    "M00-M99 diseases of the musculoskeletal system and connective tissue (โรคของระบบกล้ามเนื้อโครงร่าง และเนื้อเยื่อเกี่ยวพัน)",
    "G00-G99 diseases of the nervous system (โรคของระบบประสาท)",
    "J00-J99 diseases of the respiratory system (โรคของระบบหายใจ)",
    "L00-L99 diseases of the skin and subcutaneous tissue (โรคของผิวหนังและเนื้อเยื่อใต้ผิวหนัง)",
    "E00-E89 endocrine nutritional and metabolic diseases (โรคของต่อมไร้ท่อ โภชนาการ และเมตะบอลิซึม)",
    "V00-Y99 external causes of morbidity (สาเหตุภายนอกของการเจ็บป่วยและการตาย)",
    "Z00-Z99 factors influencing health status and contact with health services (ปัจจัยที่มีผลต่อสถานะสุขภาพและการรับบริการสุขภาพ)",
    "S00-T88 injury poisoning and certain other consequences of external causes (การบาดเจ็บ การเป็นพิษ และผลสืบเนื่องบางอย่างจากสาเหตุภายนอก)",
    "F01-F99 mental behavioral and neurodevelopmental disorders (ความผิดปกติทางจิตและพฤติกรรม)",
    "C00-D49 neoplasms (เนื้องอกและมะเร็ง)",
    "O00-O9A pregnancy childbirth and the puerperium (การตั้งครรภ์ การคลอด และระยะหลังคลอด)",
    "R00-R99 symptoms signs and abnormal clinical and laboratory findings not elsewhere classified (อาการ อาการแสดง และความผิดปกติที่พบจากการตรวจทางคลินิกและทางห้องปฏิบัติการ)",
    "unknown"
]

# Load model and tokenizer
MODEL_NAME = "transformer_model"  # Replace with your model name or path
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(class_names))

# Set the model to evaluation mode
model.eval()

def predict_single(input_text: str) -> dict:
    """
    Predict the top 3 ICD-10 categories with confidence scores for a single input text.
    If the input is empty or None, return Null with no predictions.
    """
    try:
        # Check if the input text is None or empty
        if not input_text or not input_text.strip():
            return {"input": "Null", "predicted_1": "ไม่สามารถทำนายผลได้", "predicted_2": None, "predicted_3": None,
                    "confident_1": None, "confident_2": None, "confident_3": None}
        
        # Tokenize input text
        input_text = input_text.lower().strip()
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]

        # Get top 3 predictions
        top_3 = torch.topk(probabilities, k=3)

        # Prepare results
        results = {"input": input_text}
        for i, (idx, prob) in enumerate(zip(top_3.indices.tolist(), top_3.values.tolist())):
            if idx >= len(class_names):
                raise ValueError(f"Index {idx} is out of bounds for class_names (len={len(class_names)})")
            results[f"predicted_{i+1}"] = class_names[idx]
            results[f"confident_{i+1}"] = round(prob, 4)*100

        return results

    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")
    
def predict_batch(input_file: str, output_file: str, column_name: str, language: str = "both"):
    """
    Predict the top 3 ICD-10 categories with confidence scores for a batch of texts from a CSV or XLSX file.

    Args:
        input_file (str): Path to the input file (CSV or XLSX).
        output_file (str): Path to save the output file (CSV).
        column_name (str): Column containing the text data for prediction.
        language (str): Language of the output. Options: 'thai', 'english', 'both'. Default is 'both'.
    """
    try:
        # Validate language parameter
        if language not in ["thai", "english", "both"]:
            raise ValueError("Language must be one of 'thai', 'english', or 'both'.")

        # Read input file
        if input_file.endswith(".csv"):
            df = pd.read_csv(input_file)
        elif input_file.endswith(".xlsx"):
            df = pd.read_excel(input_file)
        else:
            raise ValueError("Input file must be a CSV or XLSX.")

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the input file.")

        # Perform predictions
        predictions = []
        for text in df[column_name]:
            if pd.isnull(text) or not str(text).strip():  # Check for missing or empty input
                predictions.append({
                    "input": "Null",
                    "predicted_1": "ไม่สามารถวิเคราะห์ได้",
                    "predicted_2": None,
                    "predicted_3": None,
                    "confident_1": None,
                    "confident_2": None,
                    "confident_3": None
                })
            else:
                prediction = predict_single(str(text).strip().lower())

                # Filter predictions based on the selected language
                def format_prediction(pred, lang):
                    icd_id = pred.split(" ")[0]  # Extract ICD-10 code (e.g., "A00-B99")
                    thai_part = pred.split("(")[-1].strip(")")  # Extract Thai description
                    english_part = pred.split("(")[0].strip()  # Extract English description

                    if lang == "thai":
                        return f"{icd_id} {thai_part}"  # Add ID to Thai output
                    elif lang == "english":
                        return english_part
                    elif lang == "both":
                        return pred  # Keep the full original format

                predictions.append({
                    "input": prediction["input"],
                    "predicted_1": format_prediction(prediction["predicted_1"], language),
                    "predicted_2": format_prediction(prediction["predicted_2"], language),
                    "predicted_3": format_prediction(prediction["predicted_3"], language),
                    "confident_1": prediction["confident_1"],
                    "confident_2": prediction["confident_2"],
                    "confident_3": prediction["confident_3"]
                })

        # Add predictions to DataFrame and save
        prediction_df = pd.DataFrame(predictions)
        result_df = pd.concat([df, prediction_df], axis=1)
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    except Exception as e:
        raise ValueError(f"Error in batch prediction: {e}")
