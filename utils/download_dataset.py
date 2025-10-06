from roboflow import Roboflow

rf = Roboflow(api_key="-")   # paste your API key here
project = rf.workspace("seyyide").project("dataset-z6sm4")
dataset = project.version(5).download("yolov8")
