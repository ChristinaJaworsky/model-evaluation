from server import db
import datetime

class User(db.Model):
    __tablename__ = 'user'

    id              = db.Column(db.Integer, primary_key=True)
    created         = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    email           = db.Column(db.String())
