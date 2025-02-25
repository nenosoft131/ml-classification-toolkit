from appconfig_helper import AppConfigHelper
from fastapi import FastAPI

appconfig = AppConfigHelper(
    "MyAppConfigApp",
    "MyAppConfigEnvironment",
    "MyAppConfigProfile",
    45  # minimum interval between update checks
)

app = FastAPI()

@app.get("/some-url")
def index():
    if appconfig.update_config():
        print("New configuration received")
    # your configuration is available in the "config" attribute
    return {
        "config_info": appconfig.config
    }
