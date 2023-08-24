FROM python:slim-bookworm AS builder
RUN apt update -y

# for hnwslib (a chromdb dep): https://github.com/pypa/packaging-problems/issues/648
ARG HNSWLIB_NO_NATIVE=1
RUN apt-get install -y curl build-essential cmake libboost-all-dev git

# for unstructured: 
# https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
# https://github.com/Unstructured-IO/unstructured/blob/main/docs/source/installing.rst
RUN apt-get install -y libmagic-dev poppler-utils tesseract-ocr pandoc

# rust for safetensors, itself a dependency
RUN apt-get install -y rustc

FROM builder AS dependencies
WORKDIR /app
COPY requirements.txt /app/

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# # for gdrive integration
# RUN pip3 install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

# for gradio compatibility with docker port-forwarding
ARG GRADIO_SERVER_NAME=0.0.0.0

# # for jupyter notebook (dev use)
# # Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
# FROM dependencies AS jupyter
# RUN apt-get install tini
# RUN chmod +x /usr/bin/tini
# ENTRYPOINT ["/usr/bin/tini", "--"]
# CMD ["jupyter", "server", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

# for gradio (normal use)
CMD ["python", "code/webapp.py"]