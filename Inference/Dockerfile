# Copy to Dockerfile

# Specify the base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Copy the example directory to /app
COPY example /app

# install dependencies from requirements.txt
RUN pip install -r /app/requirements.txt

# DO NOT EDIT THE FLLOWING LINES
COPY *_run.py /
COPY submitter.json /
# You can run more commands when the container start by 
# editing docker-entrypoint.sh
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/bin/bash", "/docker-entrypoint.sh"]
