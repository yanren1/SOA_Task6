import json
from demo import onnx_inf
from api_response import respond
import base64

def lambda_handler(event, context):
    # Get image
    post_body = json.loads(event["body"])
    img_base64 = post_body.get('image')

    if img_base64 is None:
        return respond(False, None, "No image parameter received")

    try:
        b64image = base64.b64decode(img_base64)
    except Exception as e:
        print(e)
        return respond(False, None, "Couldn't decode base64 string")

    boxes, img = onnx_inf(b64image)

    return {
        "statusCode": 200,
        "body": json.dumps(respond(True, vector=img,boxes=boxes)),
        "isBase64Encoded": True
    }
