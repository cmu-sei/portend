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

FROM python:3.10

# Trusted host configs used to avoid issues when running behind SSL proxies.
RUN pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org"

# Install required opencv dependencies.
RUN apt-get update && apt-get install -y libgl1 libpq-dev gdal-bin libgdal-dev

# Set up certificates for any proxies that can get in the middle of curl/wget commands during the build
# NOTE: put any CA certificates needed for a proxy in the ./certs folder in the root of this repo, in PEM format
# but with a .crt extensions, so they can be loaded into the container and used for SSL connections properly.
RUN apt-get install -y ca-certificates
RUN mkdir /certs
COPY ./certs/*.crt /certs/
RUN if [ -n "$(ls -A /certs/*.crt)" ]; then \
      cp -rf /certs/*.crt /usr/local/share/ca-certificates/; \
      update-ca-certificates; \
    fi

# Dependencies.
WORKDIR /app
COPY requirements.txt /app/
RUN pip --default-timeout=1000 install -r requirements.txt
RUN pip --default-timeout=1000 install gdal=="`gdal-config --version`.*"

# Code.
COPY ./ /app/

ENTRYPOINT ["python", "-m"]
