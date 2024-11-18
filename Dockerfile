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

FROM python:3.9
#-alpine

# Preps needed to install Python deps.
# Install packages needed to compile some Python deps.
RUN apt-get update && apt-get install libgl1 -y

# Trusted host configs used to avoid issues when running behind SSL proxies.
RUN pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org"

# Set up certificates for any proxies that can get in the middle of curl/wget commands during the build
# NOTE: put any CA certificates needed for a proxy in the ./certs folder in the root of this repo, in PEM format
# but with a .crt extensions, so they can be loaded into the container and used for SSL connections properly.
RUN apt-get install -y ca-certificates
RUN mkdir /certs
COPY ./certs/ /certs/
RUN if [ -n "$(ls -A /certs/*.crt)" ]; then \
      cp -rf /certs/*.crt /usr/local/share/ca-certificates/; \
      update-ca-certificates; \
    fi

# Install docker client.
RUN apt-get install -y curl gnupg
RUN install -m 0755 -d /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
RUN chmod a+r /etc/apt/keyrings/docker.gpg 
RUN echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN apt-get update && \
    apt-get install -y docker-ce-cli

# Installing Python deps.
COPY pyproject.toml README.md LICENSE.txt /app/
WORKDIR /app
RUN pip --default-timeout=1000 install .

# Actual code.
COPY portend/ /app/portend/
COPY run_local.sh /app/

WORKDIR /app/
ENTRYPOINT ["bash", "run_local.sh"]
