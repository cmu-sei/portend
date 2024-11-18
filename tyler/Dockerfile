#
# Portend Toolset
# 
# Copyright 2024 Carnegie Mellon University.
# 
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
# 
# Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
# 
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
# 
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
# 
# DM24-1299
#

FROM nvcr.io/nvidia/tensorflow:22.01-tf2-py3

# Trusted host configs used to avoid issues when running behind SSL proxies.
RUN pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org"
RUN pip install pipenv

RUN pip install --upgrade pip setuptools wheel

# Various Python and C/build deps
RUN apt-get update && apt-get install -y \ 
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx
   
RUN pip install opencv-python

# Set up certificates for any proxies that can get in the middle of curl/wget commands during the build
# NOTE: put any CA certificates needed for a proxy in the ./certs folder in this folder, in PEM format
# but with a .crt extensions, so they can be loaded into the container and used for SSL connections properly.
RUN apt-get install -y ca-certificates
RUN mkdir /certs
COPY ./certs/ /certs/
RUN if [ -n "$(ls -A /certs/*.crt)" ]; then \
      cp -rf /certs/*.crt /usr/local/share/ca-certificates/; \
      update-ca-certificates; \
    fi

# create the app directory
RUN mkdir /app
WORKDIR /app

# set up the requirements
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

# Create the map cache
RUN mkdir /app/map-cache

# Create the drift inducers
COPY drifts/ /app/drifts


# Copy main app
COPY http_base/ /app/http_base
COPY help_base/ /app/help_base
COPY tyler.py /app/tyler.py

EXPOSE 6501
CMD ["python3", "tyler.py"]
#CMD ["/bin/bash"]