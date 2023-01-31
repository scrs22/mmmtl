CLASSIFICATION="classification"
DETECTION="detection"
SEGMENTATION="segmentation"

def get_method(func_name,task,environ=globals()):
    method=getattr(environ,f"{func_name}_{task}")
    return method