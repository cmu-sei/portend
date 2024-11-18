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

import os
import argparse
import time
import urllib
import numpy as np
import cv2
import drifts.image.fog.fog as fog
import drifts.image.flood.flood as flood
import drifts.image.fohis.fohis as fohis
from flask import Flask, request, jsonify, send_from_directory, send_file, redirect
from flask.logging import create_logger

DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT=6501
#MAP_TILE="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/17/49408/36427"
MAP_TILE="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{}/{}/{}"

DRIFT_TRAIN_PATH="/app/drift-train"
DRIFT_CACHE="/app/map-cache"
LEAFLET_PATH="/app/http_base"
HELP_PATH="/app/help_base"


def get_drifter(name):
    if name == "fog":
        return fog
    elif name == "flood":
        return flood
    elif name == "fohis":
        return fohis
    else:
        return None

def create_application():
    """
    Generate a flask application with needed endpoints.
    :return: app, log
    """
    # Initialize the app
    app = Flask(__name__)
    log = create_logger(app)

    start_time = time.time()

    # Debug
    env = os.getenv('ENVIRONMENT', 'production')
    app.config['ENV'] = env
    #app.config['DEBUG'] = os.getenv('DEBUG', config[env]['debug'])
    app.config['DEBUG'] = True

    app.env = env
    app.debug = True
    
    @app.route("/status")
    def index():
        """
        Return status message to indicate the build model container is running.
        :return: Status information on server in json
        """
        print("* fetch /")
        return jsonify({
            "app":"tyler",
            "environment":app.env,
            "debug":app.debug,
            "start_time":start_time
            })

    @app.route("/map")
    @app.route("/map/")
    def map_root():
        print("* fetch /map")
        return send_from_directory(LEAFLET_PATH,"index.html")

    @app.route("/map/<path:path>")
    def map(path):
         print("* fetch /map/{}".format(path))
         return send_from_directory(LEAFLET_PATH,path)

    @app.route("/drift/<path:path>")
    def drift(path):
         print("* fetch /drift/{}".format(path))
         if os.path.isdir(DRIFT_TRAIN_PATH+"/"+path):
             return send_from_directory(DRIFT_TRAIN_PATH,path+"/index.html")
         else:
             return send_from_directory(DRIFT_TRAIN_PATH,path)

    @app.route("/drift")
    def drift_root(path):
         print("* fetch /drift")
         return send_from_directory(DRIFT_TRAIN_PATH,"index.html")

    @app.route("/")
    @app.route("/help")
    @app.route("/help/")
    def help_root():
        print("* fetch /help")
        return send_from_directory(HELP_PATH,"index.html")

    @app.route("/help/<path:path>")
    def help(path):
         print("* fetch /help/{}".format(path))
         return send_from_directory(HELP_PATH,path)
    
    @app.route("/tile/<int:z>/<int:y>/<int:x>/img.png")
    @app.route("/tile/<int:z>/<int:y>/<int:x>")
    def tile(z,y,x):
        drift_name = request.args.get('drift')

        base_url = MAP_TILE.format(z,y,x)

        #
        # If the request is undrifted, just redirect.  The redirect swaps x and y compared to what 
        #
        if not drift_name or z < FLAGS.minz:
            return redirect(base_url,code=302)


        #
        # Get the drifter and the parameters for it
        #
        drifter = get_drifter(drift_name)
        params = drifter.decode_url_params(request.args)

        #
        # Make sure directory for this drift type has been created
        #
        dir_name = "%s/%d/%d/%d" % (DRIFT_CACHE,z,x,y)
        try:
            os.makedirs(dir_name)
        except:
            pass

        #
        # File (and full path) name in which to cache the data
        #
        file_name = "map-%d-%d-%d:%s:%s.png" %(z,x,y,drift_name,drifter.encode_params(params))
        path_name = dir_name+"/"+file_name

        if not os.path.exists(path_name):
            #
            # Get the raw undrifted image
            #
            print("* generating ["+drift_name+"] "+path_name) 
            req = urllib.request.urlopen(base_url)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr,-1)

            drifted_imgs = drifter.drift_images([img],params)

            #
            # Write the cached file
            #
            cv2.imwrite(path_name,drifted_imgs[0])

        return send_file(path_name, mimetype="image/png")

    return app, log

if __name__ == "__main__":
    #
    # Handle command line arguments.
    #
    parser=argparse.ArgumentParser()
    parser.add_argument("--minz",type=int,default=16,		help="Minimum zoom level to apply drift")
    FLAGS, unparsed = parser.parse_known_args()


    #
    # Create the flask application
    #
    app, logger = create_application()
    environment = os.getenv('ENVIRONMENT', 'production')

    #
    # Configure host and port
    #
    host = os.getenv('HOST', DEFAULT_HOST)
    port = os.getenv('PORT', DEFAULT_PORT)

    #
    # Log our start-up conditions
    #
    print("* starting tyler in {} environment, with debug set to {} on port {}\n".format(app.env, app.debug, port))
    logger.info("Starting tyler in {} environment, with debug set to {} on port {}\n".format(app.env, app.debug, port))

    #
    # Depending on the environment, run as a waitress server or a flask server
    #
    if environment == 'production':
        from waitress import serve
        serve(app, host=host, port=port)
        logger.debug('Application is running at http://{}:{}'.format(host, port))
    else:
        app.run(host, port)
