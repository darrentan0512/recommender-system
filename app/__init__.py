from flask import Flask
from config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions here
    
    # Register blueprints here
    from app.routes import bp
    app.register_blueprint(bp)

    return app