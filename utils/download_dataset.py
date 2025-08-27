from roboflow import Roboflow

rf = Roboflow(api_key="f8IQQJb6p0Wl8437jxAL")   # paste your API key here
project = rf.workspace("seyyide").project("dataset-z6sm4")
dataset = project.version(5).download("yolov8")
