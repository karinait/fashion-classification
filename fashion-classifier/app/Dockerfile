FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ./
RUN python3.9 -m pip install -r requirements.txt -t .

COPY app.py ./
COPY class_indices.json ./
COPY classification_model.tflite ./

ENV MODEL_PATH ./classification_model.tflite
ENV CLASSES_PATH ./class_indices.json

CMD ["app.lambda_handler"]
