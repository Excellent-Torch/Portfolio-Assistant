from mangum import Mangum
from app.main import app

# AWS Lambda handler
handler = Mangum(app, lifespan="off")