# Docker file to run a container that will run the download_wikitext.py 
# in Python 3 for Huggingface (no GPU).

# Load Python 3 image.
FROM python:3.9.12

# Set locale for variable (pulled from dockerfile in original OpenAI
# GPT2 repository).
ENV LANG=C.UTF-8

# Create a directory in the docker container. Set the working directory
# in the container to that newly created directory and then add all
# files from the current directory in the host to the working directory
# in the container.
RUN mkdir /export-wikitext
WORKDIR /export-wikitext
ADD . /export-wikitext

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.

# Install all required modules in the requirements.txt file.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

# Run the download_wikitext.py program.
CMD ["python3", "download_wikitext.py"]