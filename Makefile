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

# Automation of various common tasks

# -----------------------------------------------------------------------------
# QA
# -----------------------------------------------------------------------------

.PHONY: isort
isort:	
	isort portend/
	isort test/
	isort map_tools/

# Format all source code
.PHONY: format
format:
	black portend/
	black test/
	black map_tools/

# Lint all source code
.PHONY: lint
lint:
	flake8 portend/
	flake8 test/
	flake8 map_tools/

# Typecheck all source code
.PHONY: typecheck
typecheck:
	mypy portend/
	mypy test/
	mypy map_tools/

# All quality assurance
.PHONY: qa
qa: isort format lint typecheck

# Run unit tests with pytest
.PHONY: test
test:
	python -m pytest test
