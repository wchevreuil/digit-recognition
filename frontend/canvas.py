import sys

import streamlit as st
import requests
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
import io
from db_accessor import get_prediction_history, save_prediction, get_total_correct, get_total
import base64

st.title("MNIST Digit Recognizer")
print("rendering everything", file=sys.stderr)
with st.sidebar:
    st.write("**Draw your digit here:**")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    st.write(f"**All time accuracy:** {(get_total_correct() / get_total()):.2f}")

actual_digit = st.text_input("**True label:**", "")

if "count" not in st.session_state:
    st.session_state.count = 0
if "correct" not in st.session_state:
    st.session_state.correct = 0

if canvas_result.image_data is not None:
    # Convert the canvas image to a format the model service expects
    image = Image.fromarray((canvas_result.image_data[:, :, :3] * 255).astype(np.uint8)).convert('L')

    # Convert to numpy and find the bounding box
    image = np.array(image)
    coords = np.column_stack(np.where(image > 0))
    if coords.size > 0:  # Only proceed if there's a nonzero digit drawn
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        # Crop and resize to 28x28
        image = image[x_min:x_max, y_min:y_max]
        image = Image.fromarray(image).resize((28, 28))
    else:
        st.warning("No digit detected!")

    # Send image to FastAPI backend
    with st.form(key="predict_form"):
        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        if not actual_digit.strip():
            st.error("Please inform the digit you've drawn.")
        elif not actual_digit.isdigit():
            st.error(f"You entered: {actual_digit}. Please enter a single digit between 0 and 9.")
        else:
            # Convert image to bytes
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)

            response = requests.post(
                "http://service:8000/predict/",
                files={"file": buf.getvalue()}
            )

            if response.status_code == 200:
                result = response.json()
                st.write(f"**Predicted Label:** {result['label']}")
                st.write(f"**Confidence:** {result['confidence']:.4f}")

                st.image(f"data:image/png;base64,{result['debug_image']}", caption="Preprocessed Image Sent to Model")
                if actual_digit.isdigit():
                    is_correct = int(actual_digit) == int(result['label'])
                    if is_correct:
                        st.session_state.correct = st.session_state.correct + 1
                    st.session_state.count = st.session_state.count + 1
                    st.write(f"**Session Accuracy:** {st.session_state.correct / st.session_state.count}")
                    st.write(f"**Session Predictions:** {st.session_state.count}")
                    save_prediction(base64.b64decode(result['debug_image']), actual_digit, result['label'],
                                    result['confidence'])
                    print("Fetching updated history", file=sys.stderr)
                    st.session_state.history = get_prediction_history()

            else:
                st.error("Error: Failed to get prediction from the backend")

if "fetch_history" not in st.session_state:
    st.session_state.fetch_history = True
if st.session_state.fetch_history:
    print("Fetching history", file=sys.stderr)
    st.session_state.history = get_prediction_history()
    st.session_state.fetch_history = False

print("Printing history", file=sys.stderr)
st.write(f"**History (Last 10):**")
for img_data, true_label, predicted_label, confidence, timestamp in st.session_state.history:
    col1, col2 = st.columns([1, 2])

    with col1:
        # Convert binary image data back to PIL image
        image = Image.open(io.BytesIO(img_data))
        st.image(image, caption=f"True: {true_label}, Predicted: {predicted_label}", width=100)
        st.markdown(f"### :white_check_mark:" if true_label == predicted_label else ":x:")

    with col2:
        st.write(f"**True Label:** {true_label}")
        st.write(f"**Predicted Label:** {predicted_label}")
        st.write(f"**Confidence:** {confidence:.2f}")
        st.write(f"**Time:** {timestamp}")

    st.write("---")
