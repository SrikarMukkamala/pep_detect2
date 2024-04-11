import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
from io import BytesIO
from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)
st.set_page_config(page_title="Computer vision", page_icon="🖥️")

model = YOLO("models/best.pt")


def main():
    st.title("Object Detection App")

    choice = st.radio("Select an option", ("Upload an image", "Use webcam", "Provide image URL"))

    if choice == "Upload an image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file)

            # results = model(source=img,stream=True,classes=[0,2,3])
            # res_plotted = results[0].plot()
            results = model.track(source=frame.to_image(), verbose=False, device=gpu, stream=True, persist=True, classes=[0,2,3])
            for res in results:
                annotated_frame = res.plot()
            cv2.imwrite('images/test_image_output.jpg', res_plotted)
            
            col1, col2 = st.columns(2)

            col1.image(img, caption="Uploaded Image", use_column_width=True)
            col2.image('images/test_image_output.jpg', caption="Predected Image", use_column_width=True)

    elif choice == "Use webcam":
        client_settings = ClientSettings(
            media_stream_constraints={
                "video": True,
                "audio": False,
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )

        class ObjectDetector(VideoTransformerBase):
            def transform(self, frame):
                img = Image.fromarray(frame.to_ndarray())

                results = model(source=img)
                res_plotted = results[0].plot()
                output_frame = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

                return output_frame

        webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            client_settings=client_settings,
            video_processor_factory=ObjectDetector,
        )

    elif choice == "Provide image URL":
        image_url = st.text_input("Enter the image URL:")

        if image_url != "":
            try:
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))

                results = model(source=img,stream=True,classes=[0,2,3])
                res_plotted = results[0].plot()
                cv2.imwrite('images/test_image_output.jpg', res_plotted)
                
                col1, col2 = st.columns(2)
                col1.image(img, caption="Downloaded Image" , use_column_width=True)
                col2.image('images/test_image_output.jpg', caption="predected Image", use_column_width=True)
            except:
                st.error("Error: Invalid image URL or unable to download the image.")


if __name__ == '__main__':
    main()
