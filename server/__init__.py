from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from rq import Queue
from rq.job import Job
# from server import worker


# Set up the database
db = SQLAlchemy()

# Set up logging before creating the app
from .logging.setup_logger import setup_logging
setup_logging()

# Create the webserver app
def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    db.init_app(app)
    return app

app = create_app()
logger = app.logger


# Register APIs
# from .api.auth import auth_api
# app.register_blueprint(auth_api.blueprint, url_prefix='/api')
#
from .api.secure_endpoint_example import secure_endpoint_api
app.register_blueprint(secure_endpoint_api.blueprint, url_prefix='/api')




# Register views
from .views.index import index_view
app.register_blueprint(index_view)

# q = Queue(connection=worker.conn)
