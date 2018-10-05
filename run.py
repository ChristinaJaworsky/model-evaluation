#!flask/bin/python
import os

os.environ["LOCATION"] = "LOCAL"

from server import app, db

app.app_context().push()
app.run(debug=True, threaded=True, port=5000)

db.create_all(app)
