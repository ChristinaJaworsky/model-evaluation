import os
import configparser

parser = configparser.ConfigParser()
logging_config_location = ''

basedir = os.path.abspath(os.path.dirname(__file__))
location = os.environ.get('LOCATION')

if location == "PROD":
    parser.read(os.path.join(basedir,'config_files/prod_config.ini'))
    logging_config_location = os.path.join(basedir,'config_files/logging_config.json')

else:
    parser.read(os.path.join(basedir,'config_files/local_config.ini'))
    logging_config_location = os.path.join(basedir,'config_files/logging_config.json')



SQLALCHEMY_DATABASE_URI = 'postgresql://%(user)s:%(password)s@%(host)s:%(port)s/%(database)s' % parser['DATABASE']

SQLALCHEMY_TRACK_MODIFICATIONS = False


SECRET_KEY = parser['APP']['SECRET_KEY']
