import torch
from transformers import YolosForObjectDetection, YolosImageProcessor
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

@st.cache_resource
def load_model():
    model = YolosForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
    processor = YolosImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
    return model, processor

model, processor = load_model()


st.title("Welcome to CodeFit")
st.subheader("Dress Code Compliance Checker")
st.subheader("Allowed: Collared shirts, blouses, shirts, skirts, dresses, pants, formal shoes, sneakers")
st.subheader("Prohibited: T-shirts, shorts, short, skirts")


uploaded_file = st.file_uploader("Upload an outfit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    
    inputs = processor(images=image, return_tensors="pt")
    
  
    with torch.no_grad():
        outputs = model(**inputs)

  
    target_sizes = torch.tensor([image.size[::-1]])  
    results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

    
    allowed_items = {"Collared shirts", "Blouses", "Shirts", "Skirt", "Dress", "Pants", "Formal shoes", "Sneakers"}
    prohibited_items = {"T-shirt", "Shorts", "Short skirt"}

    detected_items = set()
    violations = set()

    draw = ImageDraw.Draw(image)

    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]  
        confidence = round(score.item() * 100, 2)

        
        box = [round(i, 2) for i in box.tolist()]
        x_min, y_min, x_max, y_max = box

       
        detected_items.add(label_name)

        
        if label_name in prohibited_items:
            violations.add(label_name)

       
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red" if label_name in prohibited_items else "green", width=3)
        draw.text((x_min, y_min), f"{label_name} ({confidence}%)", fill="white")

   
    st.image(image, caption="Detected Clothing", use_column_width=True)

    st.write(f"**Detected Clothing Items:** {detected_items}")

    if violations:
        st.error(f"❌ Prohibited Items Detected: {violations}")
    else:
        st.success("✅ Outfit is compliant with the dress code!")
