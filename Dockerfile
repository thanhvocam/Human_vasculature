FROM python:3.10

WORKDIR /app

COPY requirements.txt best.pt streamlit_app.py /app/

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

CMD [ "streamlit", "run", "streamlit_app.py" ]

EXPOSE 8501